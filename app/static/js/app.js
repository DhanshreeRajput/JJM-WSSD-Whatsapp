// Enhanced JavaScript for LangGraph Multi-Agent SQL QA System with SQL-to-NLP

// Global variables
let isLoading = false;

// Main question asking functionality
async function askQuestion() {
    const questionInput = document.getElementById('questionInput');
    const responseStyle = document.getElementById('responseStyle');
    const question = questionInput.value.trim();
    
    if (!question) {
        alert('Please enter a question.');
        return;
    }
    
    if (isLoading) {
        return;
    }
    
    isLoading = true;
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = '<div class="loading">🤔 Processing your question through our multi-agent system...</div>';
    
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                question: question,
                use_safety: true,
                response_style: responseStyle.value
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            let output = `
                <div class="result-success">
                    <h3>💡 Answer from ${result.current_agent || 'system'} agent</h3>
                    <div class="result-section">
                        <h4>📝 Response:</h4>
                        <p class="description">${result.interpretation}</p>
                    </div>
                    <div class="result-section">
                        <h4>📊 Query Details:</h4>
                        <p><strong>Rows found:</strong> ${result.row_count || 0}</p>
                        <p><strong>Execution time:</strong> ${result.execution_time?.toFixed(2) || 0}s</p>
                        <p><strong>Agent used:</strong> ${result.current_agent || 'unknown'}</p>
                        <p><strong>Safety status:</strong> ${result.is_safe ? '✅ Safe' : '❌ Unsafe'}</p>
                    </div>
            `;
            
            if (result.sql_query && result.sql_query.trim()) {
                output += `
                    <div class="result-section">
                        <h4>🔍 Generated SQL Query:</h4>
                        <pre class="analysis">${result.sql_query}</pre>
                        <button onclick="explainGeneratedSQL('${result.sql_query.replace(/'/g, "\\'")}')">
                            🔄 Explain this SQL query
                        </button>
                    </div>
                `;
            }
            
            output += '</div>';
            responseDiv.innerHTML = output;
        } else {
            responseDiv.innerHTML = `<div class="result-error">❌ Error: ${result.detail || 'Unknown error occurred'}</div>`;
        }
    } catch (error) {
        responseDiv.innerHTML = `<div class="result-error">❌ Network Error: ${error.message}</div>`;
    } finally {
        isLoading = false;
    }
}

// SQL to NLP explanation functionality
async function explainSQL() {
    const sqlQuery = document.getElementById('sqlInput').value.trim();
    const context = document.getElementById('contextInput').value.trim();
    const includeAnalysis = document.getElementById('includeAnalysis').checked;
    
    if (!sqlQuery) {
        alert('Please enter a SQL query to explain.');
        return;
    }
    
    if (isLoading) {
        return;
    }
    
    isLoading = true;
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = '<div class="loading">🔄 Converting SQL to natural language...</div>';
    
    try {
        const response = await fetch('/api/explain_sql', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sql_query: sqlQuery,
                context: context,
                include_analysis: includeAnalysis,
                language: 'en'
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            let output = `
                <div class="result-success">
                    <h3>🔄 SQL-to-NLP Conversion Result</h3>
                    <div class="result-section">
                        <h4>📝 Natural Language Description:</h4>
                        <p class="description">${result.description}</p>
                    </div>
                    <div class="result-section">
                        <h4>🔍 Query Details:</h4>
                        <p><strong>Complexity:</strong> ${result.complexity}</p>
                        <p><strong>Safety Status:</strong> ${result.is_safe ? '✅ Safe' : '❌ Unsafe'}</p>
                        <p><strong>Processed by:</strong> ${result.agent} agent</p>
                        <p><strong>Processing time:</strong> ${new Date(result.timestamp).toLocaleTimeString()}</p>
                    </div>
            `;
            
            if (result.analysis) {
                output += `
                    <div class="result-section">
                        <h4>📊 Detailed Analysis:</h4>
                        <pre class="analysis">${result.analysis}</pre>
                    </div>
                `;
            }
            
            output += '</div>';
            responseDiv.innerHTML = output;
        } else {
            responseDiv.innerHTML = `<div class="result-error">❌ Error: ${result.detail || 'Unknown error occurred'}</div>`;
        }
    } catch (error) {
        responseDiv.innerHTML = `<div class="result-error">❌ Network Error: ${error.message}</div>`;
    } finally {
        isLoading = false;
    }
}

// Explain a generated SQL query from the regular question flow
async function explainGeneratedSQL(sqlQuery) {
    if (isLoading) {
        return;
    }
    
    isLoading = true;
    const responseDiv = document.getElementById('response');
    responseDiv.innerHTML = '<div class="loading">🔄 Explaining the generated SQL query...</div>';
    
    try {
        const response = await fetch('/api/explain_sql', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sql_query: sqlQuery,
                context: "This query was generated by our system to answer a user's question",
                include_analysis: true,
                language: 'en'
            })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            let output = `
                <div class="result-success">
                    <h3>🔍 Generated SQL Query Explanation</h3>
                    <div class="result-section">
                        <h4>📝 What this query does:</h4>
                        <p class="description">${result.description}</p>
                    </div>
                    <div class="result-section">
                        <h4>🔧 Technical Details:</h4>
                        <p><strong>Complexity:</strong> ${result.complexity}</p>
                        <p><strong>Safety Status:</strong> ${result.is_safe ? '✅ Safe' : '❌ Unsafe'}</p>
                    </div>
            `;
            
            if (result.analysis) {
                output += `
                    <div class="result-section">
                        <h4>📊 Query Structure Analysis:</h4>
                        <pre class="analysis">${result.analysis}</pre>
                    </div>
                `;
            }
            
            output += '</div>';
            responseDiv.innerHTML = output;
        } else {
            responseDiv.innerHTML = `<div class="result-error">❌ Error explaining query: ${result.detail || 'Unknown error'}</div>`;
        }
    } catch (error) {
        responseDiv.innerHTML = `<div class="result-error">❌ Network Error: ${error.message}</div>`;
    } finally {
        isLoading = false;
    }
}

// System status checking
async function checkSystemStatus() {
    try {
        const response = await fetch('/health');
        const status = await response.json();
        
        updateStatusIndicator('dbStatus', status.database || 'unknown');
        updateStatusIndicator('ollamaStatus', status.llm || status.ollama_status || 'unknown');
        updateStatusIndicator('sqlToNlpStatus', status.sql_to_nlp || 'unknown');
        updateStatusIndicator('agentStatus', status.agents || 'unknown');
        
    } catch (error) {
        console.error('Status check failed:', error);
        updateStatusIndicator('dbStatus', 'error');
        updateStatusIndicator('ollamaStatus', 'error');
        updateStatusIndicator('sqlToNlpStatus', 'error');
        updateStatusIndicator('agentStatus', 'error');
    }
}

function updateStatusIndicator(elementId, status) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Remove existing classes
    element.classList.remove('status-healthy', 'status-error', 'status-unknown');
    
    // Update text and add appropriate class
    if (typeof status === 'string' && status.includes('healthy')) {
        element.textContent = '✅ Healthy';
        element.classList.add('status-healthy');
    } else if (typeof status === 'string' && status.includes('error')) {
        element.textContent = '❌ Error';
        element.classList.add('status-error');
    } else if (status === 'not_initialized') {
        element.textContent = '⚠️ Not initialized';
        element.classList.add('status-unknown');
    } else {
        element.textContent = '⚠️ Unknown';
        element.classList.add('status-unknown');
    }
}

// Sample SQL queries for testing
function loadSampleQuery(queryType) {
    const sqlInput = document.getElementById('sqlInput');
    const contextInput = document.getElementById('contextInput');
    
    const sampleQueries = {
        simple: {
            sql: "SELECT district_name FROM districts LIMIT 10",
            context: "Getting list of districts"
        },
        moderate: {
            sql: "SELECT COUNT(*) as total_grievances, status FROM grievances GROUP BY status ORDER BY total_grievances DESC",
            context: "Analyzing grievance status distribution"
        },
        complex: {
            sql: "SELECT d.district_name, COUNT(g.id) as grievance_count, AVG(EXTRACT(DAY FROM (g.resolved_date - g.created_date))) as avg_resolution_days FROM districts d LEFT JOIN grievances g ON d.id = g.district_id WHERE g.status = 'resolved' GROUP BY d.district_name HAVING COUNT(g.id) > 0 ORDER BY grievance_count DESC LIMIT 15",
            context: "Complex analysis of grievance resolution performance by district"
        }
    };
    
    const selected = sampleQueries[queryType];
    if (selected) {
        sqlInput.value = selected.sql;
        contextInput.value = selected.context;
    }
}

// Add sample query buttons
function addSampleQueryButtons() {
    const sqlSection = document.querySelector('.sql-to-nlp-section .input-group');
    if (sqlSection) {
        const sampleButtonsDiv = document.createElement('div');
        sampleButtonsDiv.innerHTML = `
            <div style="display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0;">
                <span style="font-weight: 600; color: #555;">Sample queries:</span>
                <button type="button" onclick="loadSampleQuery('simple')" style="padding: 5px 12px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; cursor: pointer;">Simple</button>
                <button type="button" onclick="loadSampleQuery('moderate')" style="padding: 5px 12px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; cursor: pointer;">Moderate</button>
                <button type="button" onclick="loadSampleQuery('complex')" style="padding: 5px 12px; border: 1px solid #ddd; border-radius: 4px; background: #f8f9fa; cursor: pointer;">Complex</button>
            </div>
        `;
        sqlSection.insertBefore(sampleButtonsDiv, sqlSection.firstChild);
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(event) {
    // Ctrl+Enter to submit question
    if (event.ctrlKey && event.key === 'Enter') {
        const activeElement = document.activeElement;
        if (activeElement.id === 'questionInput') {
            askQuestion();
        } else if (activeElement.id === 'sqlInput') {
            explainSQL();
        }
    }
});

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 LangGraph Multi-Agent SQL QA System with SQL-to-NLP loaded');
    
    // Check system status on load
    checkSystemStatus();
    
    // Set up periodic status checks
    setInterval(checkSystemStatus, 30000); // Every 30 seconds
    
    // Add sample query buttons
    setTimeout(addSampleQueryButtons, 100);
    
    // Add tooltip functionality
    addTooltips();
    
    // Focus on question input
    const questionInput = document.getElementById('questionInput');
    if (questionInput) {
        questionInput.focus();
    }
});

// Add helpful tooltips
function addTooltips() {
    const tooltips = {
        'questionInput': 'Ask natural language questions about districts, schemes, grievances, users, or tracking data',
        'sqlInput': 'Enter any SQL query to get a natural language explanation of what it does',
        'contextInput': 'Add context to help generate better explanations (optional)',
        'includeAnalysis': 'Include detailed technical analysis of the query structure'
    };
    
    Object.entries(tooltips).forEach(([elementId, tooltipText]) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.title = tooltipText;
        }
    });
}

// Error handling and retry functionality
function retryLastAction() {
    // This could be enhanced to remember the last action and retry it
    location.reload();
}

// Copy functionality for results
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard!');
    }).catch(function(err) {
        console.error('Could not copy text: ', err);
        showNotification('Copy failed', 'error');
    });
}

// Simple notification system
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 1000;
        animation: slideIn 0.3s ease;
        background: ${type === 'error' ? '#dc3545' : '#28a745'};
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .notification {
        animation: slideIn 0.3s ease;
    }
`;
document.head.appendChild(style);

// Advanced features
async function testSQLToNLPEndpoint() {
    try {
        const response = await fetch('/test_sql_to_nlp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sql_query: "SELECT * FROM districts WHERE district_name ILIKE '%Mumbai%'",
                context: "Finding Mumbai district information",
                include_analysis: true
            })
        });
        
        const result = await response.json();
        console.log('SQL-to-NLP test result:', result);
        
        if (result.status === 'success') {
            showNotification('SQL-to-NLP test successful!');
        } else {
            showNotification('SQL-to-NLP test failed', 'error');
        }
    } catch (error) {
        console.error('SQL-to-NLP test error:', error);
        showNotification('SQL-to-NLP test error', 'error');
    }
}

// Export functions for global access
window.askQuestion = askQuestion;
window.explainSQL = explainSQL;
window.explainGeneratedSQL = explainGeneratedSQL;
window.loadSampleQuery = loadSampleQuery;
window.testSQLToNLPEndpoint = testSQLToNLPEndpoint;
window.checkSystemStatus = checkSystemStatus;
window.updateStatusIndicator = updateStatusIndicator;