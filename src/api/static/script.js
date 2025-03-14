document.addEventListener("DOMContentLoaded", function() {
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-button");
    const chatHistory = document.getElementById("chat-history");
    const threadIdInput = document.getElementById("thread-id");

    function getCookie(name) {
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) return parts.pop().split(';').shift();
    }

    let threadId = getCookie("thread_id");
    if (threadId) {
        threadIdInput.value = threadId;
    }

    sendButton.addEventListener("click", function() {
        const pregunta = userInput.value;
        if (pregunta.trim() !== "") {
            agregarMensaje(pregunta, "usuario");
            enviarPregunta(pregunta, threadId);
            userInput.value = "";
        }
    });

    userInput.addEventListener("keydown", function(event) {
        if (event.key === "Enter") {
            event.preventDefault();
            sendButton.click();
        }
    });

    function agregarMensaje(mensaje, tipo) {
        const mensajeDiv = document.createElement("div");
        mensajeDiv.classList.add("mensaje");
        mensajeDiv.classList.add(tipo === "usuario" ? "mensaje-usuario" : "mensaje-bot");
        mensajeDiv.textContent = mensaje;
        chatHistory.appendChild(mensajeDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    }

    function enviarPregunta(pregunta, threadId) {
        fetch("/consultar", {
            method: "POST",
            headers: {
                "Content-Type": "application/x-www-form-urlencoded",
            },
            body: "pregunta=" + encodeURIComponent(pregunta) + "&thread_id=" + encodeURIComponent(threadId),
        })
        .then(response => response.json())
        .then(data => {
            agregarMensaje(data.respuesta, "bot");
        })
        .catch(error => {
            console.error("Error:", error);
            agregarMensaje("Hubo un error al procesar tu pregunta.", "bot");
        });
    }
});