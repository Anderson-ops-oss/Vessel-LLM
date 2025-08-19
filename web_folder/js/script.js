// DOM elements
const chatContainer = document.getElementById('chat-container');
const welcomeScreen = document.getElementById('welcome-screen');
const chatInterface = document.getElementById('chat-interface');
const questionInput = document.getElementById('question-input');
const chatQuestionInput = document.getElementById('chat-question-input');
const fileInput = document.getElementById('file-input');
const chatFileInput = document.getElementById('chat-file-input');
const sendButton = document.getElementById('send-button');
const chatSendButton = document.getElementById('chat-send-button');
const themeToggle = document.getElementById('theme-toggle');
const darkIcon = document.getElementById('dark-icon');
const lightIcon = document.getElementById('light-icon');
const fileInfo = document.getElementById('file-info');
const chatFileInfo = document.getElementById('chat-file-info');
const darkBg = document.getElementById('dark-bg');
const lightBg = document.getElementById('light-bg');
const ragToggle = document.getElementById('rag-toggle');
const ragStatus = document.getElementById('rag-status');
const fileInputLabel = document.getElementById('file-input-label');
const chatFileInputLabel = document.getElementById('chat-file-input-label');
const clearChatButton = document.getElementById('clear-chat');
const contextManagerBtn = document.getElementById('context-manager-btn');
const folderUploadArea = document.getElementById('folder-upload-area');
const folderInput = document.getElementById('folder-input');
const folderInfo = document.getElementById('folder-info');
const ragModelsBtn = document.getElementById('rag-models-btn');
const ragModelsModal = document.getElementById('rag-models-modal');
const closeRagModalBtn = document.getElementById('close-rag-modal');
const ragModelsList = document.getElementById('rag-models-list');
const modalFolderUpload = document.getElementById('modal-folder-upload');
const modalFolderInput = document.getElementById('modal-folder-input');
const modalFolderInfo = document.getElementById('modal-folder-info');
const trainRagBtn = document.getElementById('train-rag-btn');
const trainingProgressContainer = document.getElementById('training-progress-container');
const trainingProgress = document.getElementById('training-progress');
const trainingStatus = document.getElementById('training-status');
const welcomeTrainingProgress = document.getElementById('welcome-training-progress');
const welcomeTrainingProgressBar = document.getElementById('welcome-training-progress-bar');
const welcomeTrainingStatus = document.getElementById('welcome-training-status');
const chatTrainingProgress = document.getElementById('chat-training-progress');
const chatTrainingProgressBar = document.getElementById('chat-training-progress-bar');
const chatTrainingStatus = document.getElementById('chat-training-status');
const modalTrainingProgress = document.getElementById('modal-training-progress');
const modalTrainingProgressBar = document.getElementById('modal-training-progress-bar');
const modalTrainingStatus = document.getElementById('modal-training-status');
const serverLoadingOverlay = document.getElementById('server-loading-overlay');
const serverStatusMessage = document.getElementById('server-status-message');

// Initialize state variables
let isDarkMode = false; 
let isRagEnabled = false;
let currentRagModel = 'None'; 
let availableRagModels = [];
let isTraining = false; 
let isStreamingEnabled = true;
let messageCount = 0; 
let currentSessionId = localStorage.getItem('oocl_session_id') || null;
let lastTrainedModelName = ''; 
let isServerReady = false; 

// Create or get session ID
function getOrCreateSessionId() {
    if (!currentSessionId) {
        currentSessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        localStorage.setItem('oocl_session_id', currentSessionId);
    }
    return currentSessionId;
}

// Update context button display (Actually we already hide the button)
function updateContextButton() {
    if (contextManagerBtn) {
        contextManagerBtn.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 inline mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v10a2 2 0 002 2h8a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
            </svg>
            Context (${messageCount})
        `;
    }
}

// API base URL
const BASE_URL = 'http://localhost:5000';

// Initialize event listeners
function initializeEventListeners() {

    // Send button event
    sendButton?.addEventListener('click', () => sendMessage(questionInput, fileInput));
    chatSendButton?.addEventListener('click', () => sendMessage(chatQuestionInput, chatFileInput));

    // Input box Enter key event
    questionInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(questionInput, fileInput);
        }
    });
    chatQuestionInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(chatQuestionInput, chatFileInput);
        }
    });

    // File input event
    fileInput?.addEventListener('change', updateFileInfo);
    chatFileInput?.addEventListener('change', updateChatFileInfo);

    // Theme toggle
    themeToggle?.addEventListener('click', toggleTheme);

    // Clear chat
    clearChatButton?.addEventListener('click', clearChat);

    // RAG mode toggle
    ragToggle?.addEventListener('click', toggleRagMode);

    // RAG model modal events
    ragModelsBtn?.addEventListener('click', openRagModelsModal);
    closeRagModalBtn?.addEventListener('click', closeRagModelsModal);
    modalFolderUpload?.addEventListener('click', () => modalFolderInput?.click());
    modalFolderInput?.addEventListener('change', () => {
        const fileCount = modalFolderInput.files.length;
        if (modalFolderInfo) {
            modalFolderInfo.textContent = fileCount > 0 ? `${fileCount} files selected` : 'No files selected';
        }
        trainRagBtn.disabled = fileCount === 0 || isTraining;
    });
    
    trainRagBtn?.addEventListener('click', startRagTraining);

    // Add rename-related event listeners
    document.getElementById('rename-model-btn')?.addEventListener('click', showRenameInput);
    document.getElementById('use-model-btn')?.addEventListener('click', useTrainedModel);
    document.getElementById('confirm-rename-btn')?.addEventListener('click', confirmRename);
    document.getElementById('cancel-rename-btn')?.addEventListener('click', cancelRename);

    // Add SSE listener to get training progress
    const source = new EventSource(`${BASE_URL}/training_progress`);
    source.onmessage = (event) => {
        const progress = JSON.parse(event.data);

        // Only update progress if training is ongoing or there's an error
        if (progress.is_training || progress.error) {
            updateTrainingProgress(progress.percentage, progress.status, progress.error);
        } else if (progress.percentage === 0 && (!progress.status || progress.status.trim() === '')) {
            // If no training is ongoing and there's no status information, ensure the progress bar is hidden
            hideTrainingProgress();
        }

        // Hide progress bar when training is complete
        if (!progress.is_training && progress.percentage === 100 && !progress.error) {
            isTraining = false;
            setTimeout(() => {
                hideTrainingProgress();
            }, 3000);
        }
    };
    source.onerror = () => {
        console.log('SSE connection closed');
    };

    // Context manager button event (Actually hide)
    if (contextManagerBtn) {
        contextManagerBtn.addEventListener('click', manageContext);
        console.log('contextManager event listener added'); // Debug info
    } else {
        console.log('contextManagerBtn not found'); // Debug info
    }
}

// Show conversation context (Acctually hide)
async function showContext() {
    console.log('showContext function called'); // Debug info
    try {
        const sessionId = getOrCreateSessionId();
        console.log('Using session ID:', sessionId); // Debug info
        
        const response = await fetch(`${BASE_URL}/get-context?session_id=${encodeURIComponent(sessionId)}`, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();
        
        console.log('Context result:', result); // Debug info
        
        if (result.status === 'success') {
            if (result.context && result.context.length > 0) {
                const contextText = result.context.map((msg, index) => 
                    `[${index + 1}] ${msg.role.toUpperCase()}:\n${msg.content}\n${'='.repeat(50)}`
                ).join('\n\n');

                // Create a new window for context display
                const contextWindow = window.open('', '_blank', 'width=800,height=600,scrollbars=yes,resizable=yes');
                contextWindow.document.write(`
                    <html>
                        <head>
                            <title>Current Context Memory (${result.context.length} messages)</title>
                            <style>
                                body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
                                .header { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; }
                                .message { margin: 20px 0; padding: 15px; border-left: 4px solid #007bff; background-color: #f8f9fa; }
                                .user { border-left-color: #28a745; }
                                .assistant { border-left-color: #007bff; }
                                .role { font-weight: bold; color: #495057; margin-bottom: 8px; }
                                .content { white-space: pre-wrap; }
                            </style>
                        </head>
                        <body>
                            <h1 class="header">Current Context Memory</h1>
                            <p><strong>Total messages:</strong> ${result.context.length}</p>
                            <p><strong>Session ID:</strong> ${result.session_id || 'Unknown'}</p>
                            <hr>
                            ${result.context.map((msg, index) => `
                                <div class="message ${msg.role}">
                                    <div class="role">[${index + 1}] ${msg.role.toUpperCase()}</div>
                                    <div class="content">${msg.content}</div>
                                </div>
                            `).join('')}
                        </body>
                    </html>
                `);
                contextWindow.document.close();
            } else {
                alert('No conversation context available. Start a conversation to build memory.');
            }
        } else {
            alert('Failed to get context: ' + result.error);
        }
    } catch (error) {
        console.error('Error fetching context:', error);
        alert('Error fetching context: ' + error.message);
    }
}

// Clear conversation context
async function clearContext() {
    try {
        const sessionId = getOrCreateSessionId();
        console.log('Clearing context for session ID:', sessionId); // Debug info

        const response = await fetch(`${BASE_URL}/clear-context`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId })
        });
        const result = await response.json();
        
        console.log('Clear context result:', result); // Debug info
        
        if (result.status === 'success') {
            messageCount = 0;
            updateContextButton();
            displayMessage('system', result.message);
        } else {
            alert('Failed to clear context: ' + result.error);
        }
    } catch (error) {
        console.error('Error clearing context:', error);
        alert('Error clearing context: ' + error.message);
    }
}

// Manage context
async function manageContext() {
    const contextContainer = document.createElement('div');
    contextContainer.className = 'context-manager-popup fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 p-6 z-50 max-w-md w-full mx-4';
    
    contextContainer.innerHTML = `
        <div class="flex justify-between items-center mb-4">
            <h3 class="text-lg font-semibold text-gray-800 dark:text-white">Context Manager</h3>
            <button id="close-context-manager" class="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200">
                <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
            </button>
        </div>
        <div class="space-y-3">
            <button id="popup-show-context" class="w-full px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors">
                Show Current Context
            </button>
        </div>
        <p class="text-sm text-gray-500 dark:text-gray-400 mt-4">
            Current messages in context: ${messageCount}
        </p>
    `;
    
    // Add overlay
    const overlay = document.createElement('div');
    overlay.className = 'context-manager-overlay fixed inset-0 bg-black bg-opacity-50 z-40';
    
    document.body.appendChild(overlay);
    document.body.appendChild(contextContainer);
    
    // Close handlers
    const closeManager = () => {
        document.body.removeChild(contextContainer);
        document.body.removeChild(overlay);
    };
    
    document.getElementById('close-context-manager').onclick = closeManager;
    overlay.onclick = closeManager;
    
    // Button handlers
    document.getElementById('popup-show-context').onclick = async () => {
        await showContext();
        closeManager();
    };
}

// Show rename success area
function showRenameSuccessArea(modelName) {
    lastTrainedModelName = modelName;
    const renameSuccessArea = document.getElementById('rename-success-area');
    const trainedModelNameSpan = document.getElementById('trained-model-name');
    
    if (renameSuccessArea && trainedModelNameSpan) {
        trainedModelNameSpan.textContent = modelName;
        renameSuccessArea.classList.remove('hidden');
    }
}

// Show rename input area
function showRenameInput() {
    const renameInputArea = document.getElementById('rename-input-area');
    const renameInput = document.getElementById('rename-input');
    
    if (renameInputArea && renameInput) {
        renameInputArea.classList.remove('hidden');
        renameInput.focus();
    }
}

// Use trained model
function useTrainedModel() {
    if (lastTrainedModelName) {
        selectRagModel(lastTrainedModelName);
        if (!isRagEnabled) {
            toggleRagMode();
        }
    }
}

// Confirm rename
async function confirmRename() {
    const renameInput = document.getElementById('rename-input');
    const newName = renameInput?.value.trim();
    
    if (!newName) {
        alert('Please enter a valid name');
        return;
    }
    
    if (newName === lastTrainedModelName) {
        alert('New name is the same as current name');
        return;
    }
    
    try {
        const response = await fetch(`${BASE_URL}/rename_rag_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                old_name: lastTrainedModelName,
                new_name: newName
            })
        });
        
        const result = await response.json();
        if (result.success) {
            lastTrainedModelName = newName;
            showRenameSuccessArea(newName);
            displayMessage('system', `Model renamed to "${newName}" successfully`);
            loadRagModels();

            // Hide rename input area
            const renameInputArea = document.getElementById('rename-input-area');
            if (renameInputArea) {
                renameInputArea.classList.add('hidden');
            }
        } else {
            alert(`Failed to rename model: ${result.error}`);
        }
    } catch (error) {
        console.error('Error renaming model:', error);
        alert(`Error renaming model: ${error.message}`);
    }
}

// Cancel rename
function cancelRename() {
    const renameInputArea = document.getElementById('rename-input-area');
    const renameInput = document.getElementById('rename-input');
    
    if (renameInputArea) {
        renameInputArea.classList.add('hidden');
    }
    
    if (renameInput) {
        renameInput.value = '';
    }
}

// Send message function
async function sendMessage(inputElement, fileInputElement) {
    // Check if server is ready
    if (!isServerReady) {
        alert('Server is still starting up. Please wait a moment and try again.');
        return;
    }
    
    const question = inputElement.value.trim();
    // Important: Convert files to array first, as clearing fileInput will make files empty
    const files = Array.from(fileInputElement.files);
    
    // Debug message
    console.log('发送消息调试信息:');
    console.log('- 问题:', question);
    console.log('- 文件数量:', files.length);
    console.log('- RAG模式:', isRagEnabled);
    console.log('- 当前RAG模型:', currentRagModel);
    console.log('- 文件列表:', files.map(f => f.name));

    // Check whether RAG mode is enabled and files are being uploaded
    if (isRagEnabled && files.length > 0) {
        alert('File upload is disabled in RAG mode. Please disable RAG mode to upload files.');
        return;
    }

    // Check if question is empty and no files are selected
    if (!question && files.length === 0) {
        alert('Please enter a question or select files to upload.');
        return;
    }

    // Show chat interface if question input is used
    if (inputElement === questionInput) {
        showChatInterface();
    }

    // Disable input controls
    setInputControlsDisabled(true, inputElement === chatQuestionInput);

    // Show user message and clear input
    if (question) {
        displayMessage('user', question);
        inputElement.value = '';
        messageCount++;
        updateContextButton();
    }
    if (files.length > 0) {
        const fileNames = files.map(f => f.name).join(', ');
        displayMessage('user', `📎 Uploaded files: ${fileNames}`);
        fileInputElement.value = '';
        if (inputElement === questionInput) {
            updateFileInfo();
        } else {
            updateChatFileInfo();
        }
    }

    // Show thinking message
    const thinkingId = Date.now();
    displayThinkingMessage(thinkingId);

    try {
        let endpoint, formData;
        
        console.log('开始发送逻辑判断...');
        console.log('isRagEnabled:', isRagEnabled);
        console.log('files.length:', files.length);
        
        if (isRagEnabled) {
            console.log('进入RAG分支');
            endpoint = `${BASE_URL}/rag_ask`;
            const requestData = {
                question: question,
                model_id: currentRagModel,
                session_id: getOrCreateSessionId()
            };
            
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });
            
            await handleStreamingResponse(response, thinkingId);
        } else if (files.length > 0) {
            console.log('进入文件上传分支 - 准备上传文件...');
            endpoint = `${BASE_URL}/upload_and_ask`;
            formData = new FormData();
            formData.append('question', question);
            formData.append('session_id', getOrCreateSessionId());
            for (const file of files) {
                console.log(`添加文件到FormData: ${file.name}, 大小: ${file.size} bytes`);
                formData.append('file', file);
            }
            
            console.log('发送文件上传请求到:', endpoint);
            const response = await fetch(endpoint, {
                method: 'POST',
                body: formData
            });
            
            console.log('文件上传响应状态:', response.status);
            await handleStreamingResponse(response, thinkingId);
        } else {
            console.log('进入普通问答分支 - 没有文件上传');
            endpoint = `${BASE_URL}/ask-stream`;
            const requestData = {
                question: question,
                session_id: getOrCreateSessionId()
            };
            
            console.log('发送普通问答请求到:', endpoint);
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });
            
            await handleStreamingResponse(response, thinkingId);
        }
    } catch (error) {
        removeThinkingMessage(thinkingId);
        displayMessage('error', `Error: ${error.message}`);
    } finally {
        setInputControlsDisabled(false, inputElement === chatQuestionInput);
    }
}

// 处理流式响应
async function handleStreamingResponse(response, thinkingId) {
    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let aiMessageElement = null;
    let aiMessageContent = '';

    try {
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const data = JSON.parse(line.slice(6));
                        
                        if (data.error) {
                            removeThinkingMessage(thinkingId);
                            displayMessage('error', data.error);
                            return;
                        }
                        
                        if (data.content) {
                            if (!aiMessageElement) {
                                removeThinkingMessage(thinkingId);
                                aiMessageElement = createMessageElement('ai', '');
                                chatContainer.appendChild(aiMessageElement);
                            }
                            
                            aiMessageContent += data.content;
                            const contentDiv = aiMessageElement.querySelector('.message-content');
                            contentDiv.textContent = aiMessageContent;
                            scrollToBottom();
                        }
                        
                        if (data.formatted_response || data.thinking) {
                            if (aiMessageElement && data.formatted_response) {
                                const contentDiv = aiMessageElement.querySelector('.message-content');
                                contentDiv.innerHTML = data.formatted_response.replace(/\n/g, '<br>');
                            }
                            
                            if (data.thinking) {
                                displayMessage('ai', data.formatted_response || aiMessageContent, true, data.thinking);
                                if (aiMessageElement) {
                                    aiMessageElement.remove();
                                }
                            }
                        }
                        
                        if (data.done) {
                            messageCount++;
                            updateContextButton();
                            break;
                        }
                    } catch (parseError) {
                        console.error('JSON parse error:', parseError);
                    }
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}

// 更新文件信息显示
function updateFileInfo() {
    if (isRagEnabled && fileInput.files.length > 0) {
        fileInput.value = ''; // 清空文件选择
        alert('File upload is disabled in RAG mode. Please disable RAG mode to upload files.');
        return;
    }
    const files = fileInput.files;
    fileInfo.textContent = files.length > 0 ? `${files.length} file(s) selected` : '';
}

function updateChatFileInfo() {
    if (isRagEnabled && chatFileInput.files.length > 0) {
        chatFileInput.value = ''; // 清空文件选择
        alert('File upload is disabled in RAG mode. Please disable RAG mode to upload files.');
        return;
    }
    const files = chatFileInput.files;
    chatFileInfo.textContent = files.length > 0 ? `${files.length} file(s) selected` : '';
}

// 清除聊天历史
function clearChat() {
    if (confirm('Clear all chat history and reset LLM memory? This will remove all conversation context.')) {
        performClearChat();
    }
}

// 执行清除聊天的实际操作（不含确认对话框）
function performClearChat() {
    chatContainer.innerHTML = '';
    localStorage.removeItem('chatHistory');
    messageCount = 0;
    updateContextButton();
    
    // Clear backend context
    clearContext();
    
    // Show welcome screen again
    chatInterface.classList.add('hidden');
    welcomeScreen.classList.remove('hidden', 'fade-out');
    welcomeScreen.style.opacity = '1';
}

// 切换主题
function toggleTheme() {
    isDarkMode = !isDarkMode;
    document.body.classList.toggle('dark', isDarkMode);
    document.body.classList.toggle('light', !isDarkMode);
    darkIcon?.classList.toggle('hidden', !isDarkMode);
    lightIcon?.classList.toggle('hidden', isDarkMode);
    darkBg?.classList.toggle('hidden', !isDarkMode);
    lightBg?.classList.toggle('hidden', isDarkMode);
    localStorage.setItem('darkMode', isDarkMode);
    if (isDarkMode) {
        createFireflies();
    } else {
        document.querySelectorAll('.firefly').forEach(el => el.remove());
    }
}

// 切换 RAG 模式
function toggleRagMode() {
    if (!isServerReady) {
        alert('Server is still starting up. Please wait a moment and try again.');
        return;
    }
    
    if(currentRagModel === 'None') {
        alert('Please select a RAG model first by clicking the RAG Models button.');
        return;
    }
    isRagEnabled = !isRagEnabled;
    ragStatus.className = `ml-1 inline-block w-3 h-3 rounded-full ${isRagEnabled ? 'bg-green-500' : 'bg-gray-400'}`;
    
    // 更新文件输入控件的状态
    updateFileInputsState();
    
    if (isRagEnabled) {
        displayMessage('system', `RAG mode enabled with model: ${currentRagModel}. File upload is disabled in RAG mode.`);
    } else {
        displayMessage('system', 'RAG mode disabled. You can now upload files again.');
    }
}

// 更新文件输入控件的启用/禁用状态
function updateFileInputsState() {
    const fileInputs = [fileInput, chatFileInput];
    const fileLabels = [fileInputLabel, chatFileInputLabel];
    
    fileInputs.forEach(input => {
        if (input) {
            input.disabled = isRagEnabled;
            if (isRagEnabled) {
                input.value = ''; // 清空已选择的文件
            }
        }
    });
    
    fileLabels.forEach(label => {
        if (label) {
            if (isRagEnabled) {
                label.style.opacity = '0.5';
                label.style.cursor = 'not-allowed';
                label.title = 'File upload is disabled in RAG mode';
            } else {
                label.style.opacity = '1';
                label.style.cursor = 'pointer';
                label.title = '';
            }
        }
    });
    
    // 清空文件信息显示
    if (isRagEnabled) {
        if (fileInfo) fileInfo.textContent = '';
        if (chatFileInfo) chatFileInfo.textContent = '';
    }
}

// 隐藏训练进度条
function hideTrainingProgress() {
    [welcomeTrainingProgress, chatTrainingProgress, modalTrainingProgress].forEach(progress => {
        if (progress) {
            progress.classList.add('hidden');
        }
    });
}

// 打开 RAG 模型模态框
function openRagModelsModal() {
    loadRagModels();
    ragModelsModal?.classList.add('active');
}

// 关闭 RAG 模型模态框
function closeRagModelsModal() {
    ragModelsModal?.classList.remove('active');
}

// 加载 RAG 模型
async function loadRagModels() {
    try {
        const response = await fetch(`${BASE_URL}/rag_models`);
        const data = await response.json();
        availableRagModels = data.models || [];
        
        const modelsList = document.getElementById('rag-models-list');
        if (modelsList) {
            modelsList.innerHTML = '';
            if (availableRagModels.length === 0) {
                modelsList.innerHTML = '<p class="text-gray-500">No RAG models available. Train a new model below.</p>';
            } else {
                availableRagModels.forEach(model => {
                    const modelDiv = document.createElement('div');
                    modelDiv.className = `rag-model-item p-3 border rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700 ${currentRagModel === model ? 'active' : ''}`;
                    modelDiv.setAttribute('data-model', model);
                    modelDiv.innerHTML = `
                        <div class="flex justify-between items-center">
                            <span class="font-medium">${model}</span>
                            <div class="flex space-x-2">
                                <button onclick="selectRagModel('${model}')" class="px-2 py-1 bg-blue-500 text-white rounded text-xs hover:bg-blue-600">Select</button>
                                <button onclick="showRenameDialog('${model}')" class="px-2 py-1 bg-yellow-500 text-white rounded text-xs hover:bg-yellow-600">Rename</button>
                                <button onclick="showDeleteDialog('${model}')" class="px-2 py-1 bg-red-500 text-white rounded text-xs hover:bg-red-600">Delete</button>
                            </div>
                        </div>
                    `;
                    modelsList.appendChild(modelDiv);
                });
            }
        }
    } catch (error) {
        console.error('Error loading RAG models:', error);
        displayMessage('error', 'Failed to load RAG models: ' + error.message);
    }
}

// 选择 RAG 模型
function selectRagModel(modelName) {
    currentRagModel = modelName;
    
    // 更新模型列表中的选中状态
    document.querySelectorAll('.rag-model-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // 设置新选中的模型
    const selectedItem = document.querySelector(`[data-model="${modelName}"]`);
    if (selectedItem) {
        selectedItem.classList.add('active');
    }
    
    // 显示选择成功消息
    displayMessage('system', `RAG model "${modelName}" selected successfully.`);
    
    // 关闭模态框
    closeRagModelsModal();
}

// 显示重命名对话框
function showRenameDialog(modelName) {
    const newName = prompt(`Rename model "${modelName}" to:`, modelName);
    if (newName && newName.trim() && newName.trim() !== modelName) {
        renameExistingModel(modelName, newName.trim());
    }
}

// 显示删除确认对话框
function showDeleteDialog(modelName) {
    if (confirm(`Are you sure you want to delete the model "${modelName}"? This action cannot be undone.`)) {
        deleteExistingModel(modelName);
    }
}

// 重命名现有模型
async function renameExistingModel(oldName, newName) {
    try {
        const response = await fetch(`${BASE_URL}/rename_rag_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ old_name: oldName, new_name: newName })
        });
        
        const result = await response.json();
        if (result.success) {
            displayMessage('system', result.message);
            if (currentRagModel === oldName) {
                currentRagModel = newName;
            }
            loadRagModels();
        } else {
            alert(`Failed to rename model: ${result.error}`);
        }
    } catch (error) {
        console.error('Error renaming model:', error);
        alert(`Error renaming model: ${error.message}`);
    }
}

// 删除现有模型
async function deleteExistingModel(modelName) {
    try {
        const response = await fetch(`${BASE_URL}/delete_rag_model`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ model_name: modelName })
        });
        
        const result = await response.json();
        if (result.success) {
            displayMessage('system', result.message);
            if (currentRagModel === modelName) {
                currentRagModel = 'None';
                isRagEnabled = false;
                ragStatus.className = 'ml-1 inline-block w-3 h-3 rounded-full bg-gray-400';
            }
            loadRagModels();
        } else {
            alert(`Failed to delete model: ${result.error}`);
        }
    } catch (error) {
        console.error('Error deleting model:', error);
        alert(`Error deleting model: ${error.message}`);
    }
}

// 更新训练进度 UI
function updateTrainingProgress(percentage, status, isError = false) {
    // 确保显示进度条（只有在此函数被调用时才显示）
    const progressElements = [
        { container: welcomeTrainingProgress, bar: welcomeTrainingProgressBar, status: welcomeTrainingStatus },
        { container: chatTrainingProgress, bar: chatTrainingProgressBar, status: chatTrainingStatus },
        { container: modalTrainingProgress, bar: modalTrainingProgressBar, status: modalTrainingStatus }
    ];
    
    progressElements.forEach(({ container, bar, status: statusElement }) => {
        if (container && bar && statusElement) {
            container.classList.remove('hidden');
            bar.style.width = `${percentage}%`;
            statusElement.textContent = status || '';
            
            if (isError) {
                bar.classList.add('bg-red-500');
                bar.classList.remove('bg-blue-500');
            } else {
                bar.classList.add('bg-blue-500');
                bar.classList.remove('bg-red-500');
            }
        }
    });
}

// 开始 RAG 训练
async function startRagTraining() {
    if (isTraining) {
        displayMessage('error', 'A training process is already in progress.');
        return;
    }

    const files = modalFolderInput.files;
    if (files.length === 0) {
        displayMessage('error', 'Please select a folder with files.');
        return;
    }

    // 从文件路径中提取文件夹名称作为模型名称
    const firstFile = files[0];
    const pathParts = firstFile.webkitRelativePath.split('/');
    const folderName = pathParts[0];
    const modelName = folderName.replace(/[^a-zA-Z0-9_-]/g, '_'); // 清理名称

    isTraining = true;
    trainRagBtn.disabled = true;
    closeRagModelsModal();
    updateTrainingProgress(10, 'Uploading files...');

    try {
        const formData = new FormData();
        formData.append('folder_name', modelName); // 发送文件夹名称而不是用户输入的名称
        for (const file of files) {
            formData.append('files[]', file);
        }

        updateTrainingProgress(30, 'Processing files...');

        const response = await fetch(`${BASE_URL}/upload_folder_for_rag`, {
            method: 'POST',
            body: formData
        });

        updateTrainingProgress(80, 'Finalizing training...');

        const result = await response.json();
        if (result.error) {
            throw new Error(result.error);
        }

        updateTrainingProgress(100, 'Training complete!');
        displayMessage('system', `RAG model "${result.model_name}" trained successfully with ${result.processed_files} files.`);
        showRenameSuccessArea(result.model_name);

        setTimeout(() => {
            hideTrainingProgress();
            isTraining = false;
            trainRagBtn.disabled = false;
        }, 1500);
    } catch (error) {
        console.error('RAG training error:', error);
        displayMessage('error', `RAG training failed: ${error.message}`);
        updateTrainingProgress(100, `Error: ${error.message || 'Failed to train model'}`, true);
        setTimeout(() => {
            hideTrainingProgress();
            isTraining = false;
            trainRagBtn.disabled = false;
        }, 3000);
    }
}

// 创建消息元素（用于流式响应）
function createMessageElement(type, content) {
    const div = document.createElement('div');
    div.classList.add('chat-bubble', type === 'user' ? 'user-bubble' : type === 'ai' ? 'ai-bubble' : 'system-bubble');
    const header = document.createElement('div');
    header.className = 'flex items-center mb-1';
    const avatar = document.createElement('div');
    avatar.className = 'w-6 h-6 rounded-full flex items-center justify-center mr-2 text-white';
    avatar.style.backgroundColor = type === 'user' ? '#3b82f6' : type === 'ai' ? '#1e293b' : '#6b7280';
    const avatarText = document.createElement('span');
    avatarText.className = 'text-xs font-bold';
    avatarText.textContent = type === 'user' ? 'You' : type === 'ai' ? 'AI' : 'System';
    avatar.appendChild(avatarText);
    header.appendChild(avatar);
    div.appendChild(header);
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.style.whiteSpace = 'pre-wrap'; // Preserve spaces and line breaks
    if (content.includes('\n')) {
        contentDiv.innerHTML = content.replace(/\n/g, '<br>');
    } else {
        contentDiv.textContent = content;
    }
    div.appendChild(contentDiv);
    return div;
}

// 滚动到底部
function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 显示消息
function displayMessage(type, content, isThinking = false, thinkingContent = '') {
    if (isThinking) {
        const details = document.createElement('details');
        details.className = 'thinking-section';
        details.innerHTML = `
            <summary>
                <span class="thinking-arrow">▶</span>
                Thinking Process
            </summary>
            <div class="thinking-content">
                ${thinkingContent.replace(/\n/g, '<br>')}
            </div>
        `;
        chatContainer.appendChild(details);
        
        // 然后添加实际回答
        const messageElement = createMessageElement(type, content);
        chatContainer.appendChild(messageElement);
    } else {
        const messageElement = createMessageElement(type, content);
        chatContainer.appendChild(messageElement);
    }
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 保存消息到本地存储
function saveMessage(type, content, isThinking = false, thinkingContent = '') {
    const history = JSON.parse(localStorage.getItem('chatHistory')) || [];
    history.push({ type, content, isThinking, thinkingContent });
    localStorage.setItem('chatHistory', JSON.stringify(history));
}

// 显示思考中消息
function displayThinkingMessage(id) {
    const div = document.createElement('div');
    div.id = `thinking-${id}`;
    div.className = 'chat-bubble thinking-bubble';
    div.innerHTML = `
        <div class="flex items-center mb-1">
            <div class="w-6 h-6 rounded-full flex items-center justify-center mr-2 text-white bg-gray-500">
                <span class="text-xs font-bold">AI</span>
            </div>
        </div>
        <div class="flex items-center space-x-2">
            <div class="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-500"></div>
            <span>Thinking...</span>
        </div>
    `;
    chatContainer.appendChild(div);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// 移除思考中消息
function removeThinkingMessage(id) {
    const thinkingMsg = document.getElementById(`thinking-${id}`);
    if (thinkingMsg) {
        thinkingMsg.remove();
    }
}

// 显示聊天界面
function showChatInterface() {
    welcomeScreen?.classList.add('fade-out');
    setTimeout(() => {
        welcomeScreen?.classList.add('hidden');
        chatInterface?.classList.remove('hidden');
    }, 100);
}

// 启用/禁用输入控件
function setInputControlsDisabled(isDisabled, isChat) {
    if (isChat) {
        if (chatQuestionInput) chatQuestionInput.disabled = isDisabled;
        if (chatSendButton) chatSendButton.disabled = isDisabled;
        if (chatFileInput) chatFileInput.disabled = isDisabled;
    } else {
        if (questionInput) questionInput.disabled = isDisabled;
        if (sendButton) sendButton.disabled = isDisabled;
        if (fileInput) fileInput.disabled = isDisabled;
    }
}

// 检查服务器状态
async function checkServerStatus() {
    const maxRetries = 30; // 最大重试次数
    const retryDelay = 2000; // 重试间隔（毫秒）
    
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
            updateServerStatus(`Checking server status... (${attempt}/${maxRetries})`);
            
            const response = await fetch(`${BASE_URL}/health`, { timeout: 5000 });
            const data = await response.json();
            
            if (data.status === 'ok') {
                updateServerStatus('Server is running, checking LLM...');
                
                try {
                    const llmResponse = await fetch(`${BASE_URL}/test_llm`, { timeout: 10000 });
                    const llmData = await llmResponse.json();
                    
                    if (llmData.status === 'ok') {
                        updateServerStatus('All systems ready!');
                        hideServerLoadingOverlay();
                        return; // 服务器和LLM都准备就绪
                    } else {
                        updateServerStatus('LLM not ready, retrying...');
                    }
                } catch (error) {
                    updateServerStatus('LLM test failed, retrying...');
                }
            }
        } catch (error) {
            updateServerStatus(`Connection failed, retrying... (${attempt}/${maxRetries})`);
        }
        
        // 如果不是最后一次尝试，等待后重试
        if (attempt < maxRetries) {
            await new Promise(resolve => setTimeout(resolve, retryDelay));
        }
    }
    
    // 所有重试都失败了
    updateServerStatus('Failed to connect to server. Please check if the server is running.');
    showServerError();
}

// 更新服务器状态消息
function updateServerStatus(message) {
    if (serverStatusMessage) {
        serverStatusMessage.textContent = message;
    }
}

// 隐藏服务器加载遮罩
function hideServerLoadingOverlay() {
    isServerReady = true;
    if (serverLoadingOverlay) {
        serverLoadingOverlay.style.opacity = '0';
        setTimeout(() => {
            serverLoadingOverlay.style.display = 'none';
        }, 300);
    }
}

// 显示服务器错误（修改现有函数以适配新的加载状态）
function showServerError() {
    if (serverLoadingOverlay) {
        const overlay = serverLoadingOverlay;
        const content = overlay.querySelector('.text-center');
        content.innerHTML = `
            <div class="mb-4">
                <img src="img/OOCL_logo_slogan.png" alt="OOCL Logo" class="h-24 w-auto mx-auto mb-4">
            </div>
            <div class="text-red-500 mb-4">
                <svg class="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.732-.833-2.5 0L4.314 16.5c-.77.833.192 2.5 1.732 2.5z"></path>
                </svg>
                <h2 class="text-xl font-bold mb-2">Server Connection Failed</h2>
            </div>
            <p class="text-sm dark:text-gray-400 text-gray-600 max-w-md mb-4">
                Unable to connect to the AI assistant server. Please ensure the server is running and try refreshing the page.
            </p>
            <button onclick="location.reload()" class="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors">
                Retry
            </button>
        `;
    }
}

// 显示服务器错误
function displayServerError() {
    const errorMsg = document.createElement('div');
    errorMsg.className = 'p-4 mb-4 text-sm text-white bg-red-500 rounded-lg';
    errorMsg.textContent = 'Failed to connect to server. Please make sure the Flask server is running.';
    document.querySelector('.welcome-container')?.prepend(errorMsg);
}

// 显示 LLM 错误
function displayLLMError() {
    const errorMsg = document.createElement('div');
    errorMsg.className = 'p-4 mb-4 text-sm text-white bg-orange-500 rounded-lg';
    errorMsg.textContent = 'Server is running but cannot connect to LLM. Please make sure LM Studio is running.';
    document.querySelector('.welcome-container')?.prepend(errorMsg);
}

// 创建萤火虫效果
function createFireflies() {
    const darkBg = document.getElementById('dark-bg');
    const fireflyCount = 30; 

    for (let i = 0; i < fireflyCount; i++) {
        const firefly = document.createElement('div');
        firefly.className = 'firefly';
        
        // 随机位置和动画持续时间
        firefly.style.left = Math.random() * 100 + '%';
        firefly.style.top = Math.random() * 100 + '%';
        firefly.style.animationDelay = Math.random() * 200 + 's';
        firefly.style.animationDuration = (Math.random() * 100 + 100) + 's';
        
        // 为伪元素设置不同的动画延迟
        const style = document.createElement('style');
        style.textContent = `
            .firefly:nth-child(${i+1})::before { 
                animation-delay: ${Math.random() * 50}s; 
                animation-duration: ${Math.random() * 20 + 20}s; 
            }
            .firefly:nth-child(${i+1})::after { 
                animation-delay: ${Math.random() * 50}s; 
                animation-duration: ${Math.random() * 6 + 3}s; 
            }
        `;
        document.head.appendChild(style);
        
        darkBg.appendChild(firefly);
    }
}

// 初始化 UI
function initUI() {
    // 初始化主题设置
    darkIcon?.classList.toggle('hidden', !isDarkMode);
    lightIcon?.classList.toggle('hidden', isDarkMode);
    darkBg?.classList.toggle('hidden', !isDarkMode);
    lightBg?.classList.toggle('hidden', isDarkMode);
    
    // 确保加载遮罩也遵循当前主题
    if (serverLoadingOverlay) {
        serverLoadingOverlay.classList.toggle('dark', isDarkMode);
    }
    
    if (isDarkMode) {
        createFireflies();
    }
    
    // 确保训练进度条在初始化时是隐藏的
    hideTrainingProgress();
    
    // 加载聊天历史
    const history = JSON.parse(localStorage.getItem('chatHistory')) || [];
    if (history.length > 0) {
        showChatInterface();
        messageCount = history.filter(msg => msg.type === 'user').length;
        history.forEach(msg => {
            displayMessage(msg.type, msg.content, msg.isThinking, msg.thinkingContent);
        });
        updateContextButton();
    }

    // 初始化文件输入控件状态
    updateFileInputsState();

    // 最后检查服务器状态（这将管理加载遮罩的显示/隐藏）
    checkServerStatus();
}

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', () => {
    // 检测页面刷新并自动清除聊天历史
    // 使用 performance.navigation.type 或 performance.getEntriesByType 来检测刷新
    if (performance.navigation && performance.navigation.type === 1) {
        // 页面是通过刷新加载的，执行清除操作
        performClearChat();
    } else if (performance.getEntriesByType('navigation')[0]?.type === 'reload') {
        // 现代浏览器的刷新检测
        performClearChat();
    }
    initializeEventListeners();
    initUI();
});
