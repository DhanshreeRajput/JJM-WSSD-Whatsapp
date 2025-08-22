"""LangGraph orchestrator for multi-agent SQL QA system."""

import asyncio
import logging
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

from app.agents.location_agent import LocationAgent
from app.agents.user_agent import UserAgent
from app.agents.grievance_agent import GrievanceAgent
from app.agents.schemes_agent import SchemesAgent
from app.agents.tracker_agent import TrackerAgent
from app.agents.sql_to_nlp_agent import SQLToNLPAgent
from app.core.database import DatabaseManager
from app.config import settings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State shared between agents."""
    user_question: str
    original_question: str
    messages: Annotated[List[BaseMessage], "The messages in the conversation"]
    current_agent: str
    agent_response: Dict[str, Any]
    sql_query: str
    query_result: Any
    interpretation: str
    execution_time: float
    is_safe: bool
    validation_message: str
    row_count: int
    metadata: Dict[str, Any]
    error: Optional[str]
    iteration_count: int


class RouterAgent:
    """Router agent that determines which specialist agent to use."""
    
    def __init__(self, db_manager: DatabaseManager):
        """Initialize router agent."""
        self.db_manager = db_manager
        self.agent_name = "router"
        
        from langchain_community.llms import Ollama
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        # Initialize LLM
        self.llm = Ollama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE
        )
        
        # Setup routing prompt
        self.routing_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are a routing agent for the Water Supply and Sanitation Department database system.

Analyze this question and determine which specialist agent should handle it:

Available agents and their expertise:
- LOCATION: Districts, circles, blocks, villages, regions, administrative boundaries, geographic data
- USER: Citizens, users, registrations, accounts, profiles, user management  
- GRIEVANCE: Complaints, grievances, issues, problems, resolutions, complaint management
- SCHEMES: Government schemes, programs, projects, initiatives, policy implementation
- TRACKER: Tracking status, progress, logs, history, resolution tracking, assignment tracking

Question: {question}

Keywords to consider:
- Location words: district, circle, block, village, area, region, administrative, boundary, geographic
- User words: citizen, user, registration, account, profile, member, person, people
- Grievance words: complaint, grievance, issue, problem, resolve, report, ticket, concern
- Scheme words: scheme, program, project, initiative, policy, benefit, subsidy, assistance
- Tracker words: track, status, progress, log, history, timeline, assignment, resolve

Based on the primary intent and keywords, which agent should handle this?

Respond with just the agent name (location, user, grievance, schemes, or tracker):"""
        )
        
        # Response generation prompt
        self.response_prompt = PromptTemplate(
            input_variables=["question", "sql_result", "row_count"],
            template="""You are a friendly AI assistant for the Water Supply and Sanitation Department.

User asked: {question}
Data found: {sql_result}
Number of records: {row_count}

Create a natural, conversational response that:
1. Directly answers their question using the data
2. Highlights key findings in a user-friendly way
3. Uses phrases like "I found...", "Looking at your data...", "Here's what I see..."
4. Never mentions SQL, databases, or technical details
5. If multiple records, summarize key information
6. Suggests related questions they might ask
7. Be encouraging and helpful

For WSSD context, remember:
- Citizens are the people you serve
- Grievances are complaints that need resolution
- Schemes are government programs for citizen benefit
- Locations help organize service delivery
- Tracking shows progress on resolving issues

Your natural, helpful response:"""
        )
        
        # Create LLM chains
        self.routing_chain = LLMChain(llm=self.llm, prompt=self.routing_prompt)
        self.response_chain = LLMChain(llm=self.llm, prompt=self.response_prompt)
        
        logger.info("Router agent initialized")
    
    async def process(self, question: str) -> Dict[str, Any]:
        """Determine which agent should handle the question."""
        try:
            # Get routing decision
            routing_response = await asyncio.to_thread(
                self.routing_chain.run,
                question=question
            )
            
            # Parse the response to determine target agent
            target_agent = self._parse_routing_decision(routing_response)
            
            return {
                "agent": "router",
                "target_agent": target_agent,
                "routing_reason": routing_response,
                "question": question
            }
            
        except Exception as e:
            logger.error(f"Router agent error: {e}")
            return {
                "agent": "router",
                "target_agent": "location",  # Default fallback
                "routing_reason": f"Error in routing: {str(e)}",
                "question": question
            }
    
    def _parse_routing_decision(self, routing_response: str) -> str:
        """Parse the routing decision from LLM response."""
        response_lower = routing_response.lower().strip()
        
        # Direct agent name matches
        if "location" in response_lower:
            return "location"
        elif "user" in response_lower:
            return "user"
        elif "grievance" in response_lower:
            return "grievance"
        elif "schemes" in response_lower:
            return "schemes"
        elif "tracker" in response_lower:
            return "tracker"
        
        # Fallback keyword analysis
        agent_keywords = {
            "location": ["district", "circle", "block", "village", "area", "region", "administrative"],
            "user": ["citizen", "user", "registration", "account", "profile", "people"],
            "grievance": ["complaint", "grievance", "issue", "problem", "resolve", "report"],
            "schemes": ["scheme", "program", "project", "initiative", "benefit"],
            "tracker": ["track", "status", "progress", "log", "history", "timeline"]
        }
        
        # Score each agent based on keyword matches
        agent_scores = {}
        for agent, keywords in agent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in response_lower)
            agent_scores[agent] = score
        
        # Return agent with highest score, default to location
        if max(agent_scores.values()) == 0:
            return "location"
        
        return max(agent_scores, key=agent_scores.get)
    
    async def generate_response(
        self, 
        question: str, 
        query_result: Any, 
        row_count: int
    ) -> Dict[str, Any]:
        """Generate final user-friendly response."""
        try:
            interpretation = await asyncio.to_thread(
                self.response_chain.run,
                question=question,
                sql_result=str(query_result)[:1500] if query_result else "No data found",
                row_count=row_count
            )
            
            return {
                "interpretation": interpretation.strip(),
                "formatted_result": self._format_result_for_display(query_result, row_count)
            }
            
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return {
                "interpretation": "I found some information for you, but had trouble formatting the response.",
                "formatted_result": str(query_result) if query_result else "No data available"
            }
    
    def _format_result_for_display(self, result: Any, row_count: int) -> str:
        """Format query result for user-friendly display."""
        if not result:
            return "No data found."
        
        if row_count == 1:
            return f"Found 1 record: {result}"
        elif row_count <= 5:
            return f"Found {row_count} records: {result}"
        else:
            return f"Found {row_count} records. Showing first few: {str(result)[:500]}..."


class LangGraphSQLQA:
    """LangGraph-based multi-agent SQL QA system."""
    
    def __init__(self, database_uri: str = None):
        """Initialize the LangGraph SQL QA system."""
        self.database_uri = database_uri or settings.database_url
        self.startup_time = datetime.now()
        
        # Initialize database manager
        self.db_manager = DatabaseManager(self.database_uri)
        
        # Initialize agents (including SQL to NLP agent)
        self.agents = self._initialize_agents()
        
        # Create and compile the graph
        self.graph = self._create_graph()
        self.app = self.graph.compile(checkpointer=MemorySaver())
        
        logger.info("LangGraph SQL QA system initialized successfully with SQL-to-NLP capabilities")
    
    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents."""
        agents = {
            "router": RouterAgent(self.db_manager),
            "location": LocationAgent(self.db_manager),
            "user": UserAgent(self.db_manager),
            "grievance": GrievanceAgent(self.db_manager),
            "schemes": SchemesAgent(self.db_manager),
            "tracker": TrackerAgent(self.db_manager),
            "sql_to_nlp": SQLToNLPAgent(self.db_manager)  # Add SQL to NLP agent
        }
        
        logger.info(f"Initialized {len(agents)} agents (including SQL-to-NLP)")
        return agents
    
    def _create_graph(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for each agent
        workflow.add_node("router", self._router_node)
        workflow.add_node("location", self._location_node)
        workflow.add_node("user", self._user_node)
        workflow.add_node("grievance", self._grievance_node)
        workflow.add_node("schemes", self._schemes_node)
        workflow.add_node("tracker", self._tracker_node)
        workflow.add_node("response_generator", self._response_generator_node)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional edges from router
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "location": "location",
                "user": "user", 
                "grievance": "grievance",
                "schemes": "schemes",
                "tracker": "tracker"
            }
        )
        
        # All specialist agents flow to response generator
        for agent in ["location", "user", "grievance", "schemes", "tracker"]:
            workflow.add_edge(agent, "response_generator")
        
        # Response generator ends the flow
        workflow.add_edge("response_generator", END)
        
        return workflow
    
    async def _router_node(self, state: AgentState) -> AgentState:
        """Router agent node."""
        try:
            result = await self.agents["router"].process(state["user_question"])
            state["current_agent"] = result.get("target_agent", "location")
            state["agent_response"] = result
            state["messages"].append(AIMessage(content=f"Routing to {state['current_agent']} agent"))
            logger.info(f"Router decision: {state['current_agent']} agent")
            return state
        except Exception as e:
            logger.error(f"Router agent error: {e}")
            state["error"] = str(e)
            state["current_agent"] = "location"  # Default fallback
            return state
    
    async def _location_node(self, state: AgentState) -> AgentState:
        """Location agent node."""
        try:
            result = await self.agents["location"].process(state["user_question"])
            state["agent_response"] = result
            state["sql_query"] = result.get("sql_query", "")
            state["query_result"] = result.get("result")
            state["row_count"] = result.get("row_count", 0)
            state["is_safe"] = result.get("is_safe", True)
            state["validation_message"] = result.get("validation_message", "")
            logger.info(f"Location agent processed query, found {state['row_count']} rows")
            return state
        except Exception as e:
            logger.error(f"Location agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _user_node(self, state: AgentState) -> AgentState:
        """User agent node."""
        try:
            result = await self.agents["user"].process(state["user_question"])
            state["agent_response"] = result
            state["sql_query"] = result.get("sql_query", "")
            state["query_result"] = result.get("result")
            state["row_count"] = result.get("row_count", 0)
            state["is_safe"] = result.get("is_safe", True)
            state["validation_message"] = result.get("validation_message", "")
            logger.info(f"User agent processed query, found {state['row_count']} rows")
            return state
        except Exception as e:
            logger.error(f"User agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _grievance_node(self, state: AgentState) -> AgentState:
        """Grievance agent node."""
        try:
            result = await self.agents["grievance"].process(state["user_question"])
            state["agent_response"] = result
            state["sql_query"] = result.get("sql_query", "")
            state["query_result"] = result.get("result")
            state["row_count"] = result.get("row_count", 0)
            state["is_safe"] = result.get("is_safe", True)
            state["validation_message"] = result.get("validation_message", "")
            logger.info(f"Grievance agent processed query, found {state['row_count']} rows")
            return state
        except Exception as e:
            logger.error(f"Grievance agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _schemes_node(self, state: AgentState) -> AgentState:
        """Schemes agent node."""
        try:
            result = await self.agents["schemes"].process(state["user_question"])
            state["agent_response"] = result
            state["sql_query"] = result.get("sql_query", "")
            state["query_result"] = result.get("result")
            state["row_count"] = result.get("row_count", 0)
            state["is_safe"] = result.get("is_safe", True)
            state["validation_message"] = result.get("validation_message", "")
            logger.info(f"Schemes agent processed query, found {state['row_count']} rows")
            return state
        except Exception as e:
            logger.error(f"Schemes agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _tracker_node(self, state: AgentState) -> AgentState:
        """Tracker agent node."""
        try:
            result = await self.agents["tracker"].process(state["user_question"])
            state["agent_response"] = result
            state["sql_query"] = result.get("sql_query", "")
            state["query_result"] = result.get("result")
            state["row_count"] = result.get("row_count", 0)
            state["is_safe"] = result.get("is_safe", True)
            state["validation_message"] = result.get("validation_message", "")
            logger.info(f"Tracker agent processed query, found {state['row_count']} rows")
            return state
        except Exception as e:
            logger.error(f"Tracker agent error: {e}")
            state["error"] = str(e)
            return state
    
    async def _response_generator_node(self, state: AgentState) -> AgentState:
        """Generate final user-friendly response."""
        try:
            if state.get("error"):
                state["interpretation"] = "I'm sorry, I encountered an issue while processing your request. Please try asking in a different way."
                state["is_safe"] = False
                state["validation_message"] = state["error"]
                return state
            
            # Use the router agent to generate final interpretation
            final_result = await self.agents["router"].generate_response(
                state["user_question"],
                state.get("query_result"),
                state.get("row_count", 0)
            )
            
            state["interpretation"] = final_result.get("interpretation", "I found some information for you.")
            state["is_safe"] = True
            state["validation_message"] = "Query processed successfully"
            
            return state
        except Exception as e:
            logger.error(f"Response generator error: {e}")
            state["interpretation"] = "I encountered an issue generating the response."
            state["error"] = str(e)
            return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Determine which agent to route to based on router decision."""
        return state.get("current_agent", "location")
    
    async def answer_question(
        self,
        question: str,
        use_safety: bool = True,
        limit_results: Optional[int] = None,
        response_style: str = "brief"
    ) -> Dict[str, Any]:
        """Process a question through the LangGraph workflow."""
        start_time = datetime.now()
        
        try:
            # Initialize state
            initial_state = AgentState(
                user_question=question,
                original_question=question,
                messages=[HumanMessage(content=question)],
                current_agent="router",
                agent_response={},
                sql_query="",
                query_result=None,
                interpretation="",
                execution_time=0.0,
                is_safe=True,
                validation_message="",
                row_count=0,
                metadata={
                    "use_safety": use_safety,
                    "limit_results": limit_results,
                    "response_style": response_style,
                    "start_time": start_time
                },
                error=None,
                iteration_count=0
            )
            
            # Run the workflow
            config = {"configurable": {"thread_id": f"thread_{datetime.now().timestamp()}"}}
            final_state = await self.app.ainvoke(initial_state, config=config)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "question": question,
                "sql_query": final_state.get("sql_query", ""),
                "result": final_state.get("query_result"),
                "interpretation": final_state.get("interpretation", ""),
                "execution_time": execution_time,
                "is_safe": final_state.get("is_safe", False),
                "validation_message": final_state.get("validation_message", ""),
                "row_count": final_state.get("row_count", 0),
                "response_style": response_style,
                "timestamp": datetime.now(),
                "current_agent": final_state.get("current_agent", "unknown")
            }
            
        except Exception as e:
            logger.error(f"Error in LangGraph workflow: {e}")
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "question": question,
                "sql_query": "",
                "result": None,
                "interpretation": "I'm sorry, I encountered an error processing your request.",
                "execution_time": execution_time,
                "is_safe": False,
                "validation_message": str(e),
                "row_count": 0,
                "response_style": response_style,
                "timestamp": datetime.now(),
                "current_agent": "error"
            }
    
    async def health_check(self) -> Dict[str, str]:
        """Perform health check of all system components."""
        health_status = {
            "database": "unknown",
            "agents": "unknown",
            "sql_to_nlp": "unknown",
            "overall": "unknown"
        }
        
        try:
            # Check database
            await self.db_manager.health_check()
            health_status["database"] = "healthy"
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
        
        try:
            # Check agents by testing router
            router_health = await self.agents["router"].generate_response("test", [], 0)
            health_status["agents"] = "healthy" if router_health else "error"
        except Exception as e:
            health_status["agents"] = f"error: {str(e)}"
        
        try:
            # Check SQL to NLP agent
            sql_to_nlp_health = await self.agents["sql_to_nlp"].health_check()
            health_status["sql_to_nlp"] = "healthy" if sql_to_nlp_health else "error"
        except Exception as e:
            health_status["sql_to_nlp"] = f"error: {str(e)}"
        
        # Overall status
        core_services = [health_status["database"], health_status["agents"]]
        if all("healthy" in status for status in core_services):
            health_status["overall"] = "healthy"
        else:
            health_status["overall"] = "degraded"
        
        return health_status
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds."""
        return (datetime.now() - self.startup_time).total_seconds()
    
    def get_table_info(self):
        """Get table information from database manager."""
        return self.db_manager.get_table_info()
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent usage and performance."""
        try:
            return {
                "total_agents": len(self.agents),
                "available_agents": list(self.agents.keys()),
                "agent_specializations": {
                    "router": "Question routing and response generation",
                    "location": "Districts, circles, blocks, villages, administrative boundaries",
                    "user": "Citizens, registrations, accounts, user management",
                    "grievance": "Complaints, grievances, issues, resolutions",
                    "schemes": "Government schemes, programs, initiatives",
                    "tracker": "Status tracking, progress monitoring, logs",
                    "sql_to_nlp": "SQL query to natural language conversion"
                },
                "system_uptime": self.get_uptime(),
                "last_health_check": datetime.now().isoformat(),
                "sql_to_nlp_features": {
                    "conversion": "SQL queries to natural language descriptions",
                    "analysis": "Query component analysis and complexity assessment",
                    "batch_processing": "Multiple query conversion support",
                    "safety_validation": "Security checks for all queries"
                }
            }
        except Exception as e:
            logger.error(f"Error getting agent statistics: {e}")
            return {"error": str(e)}

    async def convert_sql_to_nlp(
        self,
        sql_query: str,
        context: str = "",
        include_analysis: bool = False
    ) -> Dict[str, Any]:
        """Convert SQL query to natural language using the SQL-to-NLP agent."""
        try:
            sql_to_nlp_agent = self.agents["sql_to_nlp"]
            result = await sql_to_nlp_agent.convert_sql_to_nlp(
                sql_query=sql_query,
                context=context,
                include_analysis=include_analysis
            )
            return result
        except Exception as e:
            logger.error(f"Error in SQL to NLP conversion: {e}")
            return {
                "agent": "sql_to_nlp",
                "sql_query": sql_query,
                "description": "Error processing SQL query",
                "is_safe": False,
                "analysis": None,
                "error": str(e)
            }

    async def batch_convert_sql_to_nlp(
        self,
        sql_queries: List[str],
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """Convert multiple SQL queries to natural language descriptions."""
        try:
            sql_to_nlp_agent = self.agents["sql_to_nlp"]
            results = await sql_to_nlp_agent.batch_convert(sql_queries, context)
            return results
        except Exception as e:
            logger.error(f"Error in batch SQL to NLP conversion: {e}")
            return [{"error": str(e)} for _ in sql_queries]
    
    async def process_batch_questions(
        self,
        questions: List[str],
        use_safety: bool = True,
        response_style: str = "brief"
    ) -> List[Dict[str, Any]]:
        """Process multiple questions through the workflow concurrently."""
        try:
            # Limit concurrency to avoid overwhelming the system
            semaphore = asyncio.Semaphore(3)
            
            async def process_single_question(question: str):
                async with semaphore:
                    return await self.answer_question(
                        question=question,
                        use_safety=use_safety,
                        response_style=response_style
                    )
            
            # Process all questions concurrently
            tasks = [process_single_question(q) for q in questions]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions in results
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Error processing question {i}: {result}")
                    processed_results.append({
                        "question": questions[i],
                        "error": str(result),
                        "is_safe": False,
                        "execution_time": 0.0,
                        "timestamp": datetime.now()
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return [{"error": str(e)} for _ in questions]
    
    def get_workflow_visualization(self) -> Dict[str, Any]:
        """Get a visualization of the workflow for debugging."""
        try:
            return {
                "nodes": [
                    {"id": "router", "type": "router", "description": "Question routing"},
                    {"id": "location", "type": "specialist", "description": "Location queries"},
                    {"id": "user", "type": "specialist", "description": "User queries"},
                    {"id": "grievance", "type": "specialist", "description": "Grievance queries"},
                    {"id": "schemes", "type": "specialist", "description": "Schemes queries"},
                    {"id": "tracker", "type": "specialist", "description": "Tracker queries"},
                    {"id": "response_generator", "type": "generator", "description": "Response generation"}
                ],
                "edges": [
                    {"from": "START", "to": "router"},
                    {"from": "router", "to": "location", "condition": "location_query"},
                    {"from": "router", "to": "user", "condition": "user_query"},
                    {"from": "router", "to": "grievance", "condition": "grievance_query"},
                    {"from": "router", "to": "schemes", "condition": "schemes_query"},
                    {"from": "router", "to": "tracker", "condition": "tracker_query"},
                    {"from": "location", "to": "response_generator"},
                    {"from": "user", "to": "response_generator"},
                    {"from": "grievance", "to": "response_generator"},
                    {"from": "schemes", "to": "response_generator"},
                    {"from": "tracker", "to": "response_generator"},
                    {"from": "response_generator", "to": "END"}
                ],
                "flow_description": "Question → Router → Specialist Agent → Response Generator → End"
            }
        except Exception as e:
            logger.error(f"Error generating workflow visualization: {e}")
            return {"error": str(e)}