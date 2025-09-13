document.addEventListener("DOMContentLoaded", () =>  {
    const chatBox   = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn   = document.getElementById('send-btn');

    const sendMessage = async () => {
        const userMessage = userInput.value.trim();
        if (userMessage === "") return;

        appendMessage(userMessage, "user");
        userInput.value = "";

        const loadingIndicator = appendMessage("...", 'arka');

        try {
            const response = await fetch('http://127.0.0.1:5000/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ message: userMessage})
            });

            if (!response.ok) throw new Error('Server response was not ok.');
            const data = await response.json();

            const arkaMessageElement = loadingIndicator.querySelector('p');
            arkaMessageElement.innerHTML = marked.parse(data.response);
        } catch (error) {
            const errorElement = loadingIndicator.querySelector('p');
            errorElement.textContent = 'Maaf, terjadi kesalahan. Coba lagi.';
            console.error('Error', error);
        }
    };

    const appendMessage = (message, sender) => {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('chat-message', sender);
        const p = document.createElement('p');

        if (sender === 'user')
        {
            p.textContent = message;
        }else 
        {
            p.textContent = message;
        }
        
        msgDiv.appendChild(p);
        chatBox.appendChild(msgDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
        return msgDiv;
    };

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });
});