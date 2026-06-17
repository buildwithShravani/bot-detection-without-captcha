// -------------------------------
// GLOBAL BEHAVIOR VARIABLES
// -------------------------------
let mouseMoves = 0;
let clicks = 0;
let maxScroll = 0;
let startTime = Date.now();

let keyTimings = [];
let lastKeyTime = null;

// -------------------------------
// TRACK USER BEHAVIOR
// -------------------------------
document.addEventListener("mousemove", () => mouseMoves++);
document.addEventListener("click", () => clicks++);

document.addEventListener("scroll", () => {
    maxScroll = Math.max(maxScroll, window.scrollY);
});

document.addEventListener("keydown", () => {
    const now = Date.now();
    if (lastKeyTime !== null) {
        keyTimings.push(now - lastKeyTime);
    }
    lastKeyTime = now;
});

// -------------------------------
// SIGNUP FUNCTION
// -------------------------------
function signup() {
    const email = document.getElementById("signupEmail").value;
    const password = document.getElementById("signupPassword").value;

    if (!email || !password) {
        alert("Please fill all fields");
        return;
    }

    localStorage.setItem(email, password);
    alert("Account created successfully!");
    window.location.href = "index.htm";
}

// -------------------------------
// LOGIN FUNCTION (FINAL FIXED)
// -------------------------------
    function login() {

    const email = document.getElementById("loginEmail").value;
    const password = document.getElementById("loginPassword").value;

    const storedPassword = localStorage.getItem(email);

    if (storedPassword !== password) {
        alert("Invalid login");
        return;
    }

    const payload = {
        mouse_moves: 500,
        clicks: 20,
        avg_typing_speed: 200,
        scroll_depth: 1000,
        time_spent: 50,
        text: email,
        typing_seq: [100,120,110,130,125,115,140,135,120,110]
    };

    // 🔥 VERY IMPORTANT CHANGE
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(res => {
        if (!res.ok) {
            throw new Error("Server error");
        }
        return res.json();
    })
    .then(data => {
        if(data.error){
            alert("Backend Error: " + data.error);
            return;
        }

        alert(
            "Result: " + data.result +
            "\nConfidence: " + data.confidence +
            "\nAction: " + data.action
        );

        if (data.action === "ALLOW") {
            window.location.href = "dashboard.htm";
        }

    })
    .catch(error => {
        alert("REAL ERROR: " + error.message);
        console.log(error);
    });
}

    // Prepare typing sequence
    const typingSeq = keyTimings.slice(-10);
    while (typingSeq.length < 10) typingSeq.unshift(0);

    const avgTypingSpeed =
        typingSeq.length > 0
            ? typingSeq.reduce((a, b) => a + b, 0) / typingSeq.length
            : 0;

    // Payload
    const payload = {
        mouse_moves: mouseMoves,
        clicks: clicks,
        avg_typing_speed: avgTypingSpeed,
        scroll_depth: maxScroll,
        time_spent: Math.round((Date.now() - startTime) / 1000),
        text: email,
        typing_seq: typingSeq
    };

    // Call backend
    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {

        alert(
            "Result: " + data.result +
            "\nConfidence: " + data.confidence +
            "\nAction: " + data.action
        );

        if (data.action === "ALLOW") {
            window.location.href = "dashboard.htm";
        }

    })
    .catch(err => {
        console.error(err);
        alert("Backend not reachable ❌");
    });

// -------------------------------
// LOGOUT FUNCTION
// -------------------------------
function logout() {
    window.location.href = "index.htm";
}
