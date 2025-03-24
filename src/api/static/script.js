document.addEventListener("DOMContentLoaded", () => {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const chatHistory = document.getElementById("chat-history");
    const resetButton = document.getElementById("reset-button"); // Botón de reinicio

    // Función para obtener cookies
    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }

    // Obtener thread_id de la cookie o de la variable global
    const threadId = getCookie("thread_id");
    console.log("Thread ID obtenido:", threadId);
    
    if (!threadId) {
        console.error("¡Advertencia! thread_id no está definido");
    }

    // Función para agregar mensajes al historial
    function addMessageToChat(message, role, sources) {
        const messageContainer = document.createElement("div");
        messageContainer.classList.add("message-container");
        messageContainer.classList.add(
            role === "user" ? "user-message" : "bot-message"
        );

        const messageBubble = document.createElement("div");
        messageBubble.classList.add("message-bubble");

        const messageIcon = document.createElement("i");
        messageIcon.classList.add("fas", "message-icon");
        messageIcon.classList.add(role === "user" ? "fa-user" : "fa-robot");

        const messageText = document.createElement("p");
        messageText.textContent = message;

        messageBubble.appendChild(messageIcon);
        messageBubble.appendChild(messageText);
        messageContainer.appendChild(messageBubble);

        if (sources && sources.length > 0) {
            const sourcesDiv = document.createElement("div");
            sourcesDiv.classList.add("message-sources");
            sourcesDiv.innerHTML = `<small>Fuentes: ${sources.join(", ")}</small>`;
            messageContainer.appendChild(sourcesDiv);
        }

        chatHistory.appendChild(messageContainer);
        chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll
    }

    // Función para enviar mensajes al backend
    const sendMessage = async (message) => {
        if (!message.trim()) return;

        addMessageToChat(message, "user");

        const formData = new FormData();
        formData.append('user_message', message);
        formData.append('thread_id', threadId);
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // Debug: Mostrar más detalles del error
                const errorText = await response.text();
                console.error('Error Response:', {
                    status: response.status,
                    statusText: response.statusText,
                    body: errorText
                });
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const html = await response.text();

            // Insertar la respuesta HTML *directamente*
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = html;
            const agentResponseDiv = tempDiv.firstChild;

            // Extraer el mensaje y las fuentes del HTML
            const messageText = agentResponseDiv.querySelector('.message-bubble p')?.textContent;
            const sourcesText = agentResponseDiv.querySelector('.message-sources small')?.textContent;
            const sources = sourcesText ? sourcesText.replace('Fuentes: ', '').split(', ') : [];

            if (messageText) {
                addMessageToChat(messageText, "bot", sources);
            }
        } catch (error) {
            console.error('Error:', error);
            addMessageToChat("Hubo un error al procesar tu pregunta.", "bot");
        }

        userInput.value = "";
    };

    // Event listeners
    sendButton.addEventListener("click", () => sendMessage(userInput.value));
    userInput.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            sendMessage(userInput.value);
        }
    });

    // Botón de reinicio
    resetButton.addEventListener("click", () => {
        const formData = new FormData();
        formData.append('thread_id', threadId);

        fetch("/api/reset", {
            method: "POST",
            body: formData
        })
        .then((response) => response.json())
        .then((data) => {
            if (data.message === "Conversación reiniciada") {
                chatHistory.innerHTML = "";
                addMessageToChat("¡Conversación reiniciada!", "bot");
            }
        })
        .catch((error) => console.error("Error:", error));
    });

    // Añadir esta función de prueba
    async function testEndpoint() {
        const formData = new FormData();
        formData.append('user_message', 'test message');
        formData.append('thread_id', threadId);

        try {
            const response = await fetch('/debug-chat', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            console.log('Debug response:', data);
        } catch (error) {
            console.error('Debug error:', error);
        }
    }

    // Llamar a esta función cuando se cargue la página
    console.log('Testing endpoint...');
    testEndpoint();
});