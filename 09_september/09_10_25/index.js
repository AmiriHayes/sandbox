import { initializeApp } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-app.js";
import { getAuth, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-auth.js";
import { getFirestore, doc, getDoc, collection, query, orderBy, getDocs } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-firestore.js";

const firebaseConfig = {
    apiKey: "AIzaSyBiPZ-0U1CZHbGzn8JS2cFCPMVK_5ftHyc",
    authDomain: "quote-app-a8c50.firebaseapp.com",
    projectId: "quote-app-a8c50",
    storageBucket: "quote-app-a8c50.firebasestorage.app",
    messagingSenderId: "1011977846276",
    appId: "1:1011977846276:web:d4a30ccef5556fe3713f51",
    measurementId: "G-2C7W5JCJDD"
};

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);

let currentQuotes = [];
let currentIndex = 0;
let userColor = "#0077cc";
let userName = "Unknown User";
let groupName = "Group";
let groupId = "group-placeholder";

// ------------------------
// Render a quote at index
// ------------------------
function renderQuoteAt(index) {
    if (currentQuotes.length === 0) return;
    const quote = currentQuotes[index];
    document.querySelector(".header").style.backgroundColor = userColor;
    document.querySelector(".quote-title").innerText = groupName;
    document.querySelector(".user-quote").innerText =
        new Date(quote.timestamp.toDate()).toLocaleDateString() + " shared by " + userName;
    document.querySelector(".user-quote").style.color = userColor;
    document.querySelector(".quote-text").innerText = "\n \"" + quote.text + "\"\n\n";
    document.querySelector(".quote button").innerText = "More about " + quote.author;
    document.querySelector(".quote button").style.backgroundColor = userColor;
    document.querySelector(".quote div:nth-child(2)").innerText = `— ${quote.author}, ${quote.source}`;
    document.querySelector(".quote-footer").innerText = `- ${groupName} Quote #${index + 1} -`;
    document.querySelector(".latest-label").style.display = "inline";
}

// ------------------------
// Load all quotes
// ------------------------
async function loadQuotes() {
    const q = query(collection(db, "groups", groupId, "quotes"), orderBy("timestamp", "asc"));
    const snap = await getDocs(q);
    currentQuotes = snap.docs.map(doc => doc.data());
    console.log("Loaded quotes:", currentQuotes);
    currentIndex = currentQuotes.length - 1;
    renderQuoteAt(currentIndex);
}

// ------------------------
// Setup buttons safely
// ------------------------
document.addEventListener("DOMContentLoaded", () => {
    const leftArrow = document.querySelector(".leftarrow");
    const rightArrow = document.querySelector(".rightarrow");
    const addNew = document.querySelector(".addnew");
    const returnHome = document.querySelector(".returnhome");

    leftArrow.addEventListener("click", () => {
        if (currentQuotes.length === 0) return;
        currentIndex = (currentIndex - 1 + currentQuotes.length) % currentQuotes.length;
        renderQuoteAt(currentIndex);
    });

    rightArrow.addEventListener("click", () => {
        if (currentQuotes.length === 0) return;
        currentIndex = (currentIndex + 1) % currentQuotes.length;
        renderQuoteAt(currentIndex);
    });

    addNew.addEventListener("click", () => {
        // addNewQuote(); // implement later
        alert("Add new quote button clicked!");
    });

    returnHome.addEventListener("click", () => {
        window.location.href = "index.html";
    });
});

// ------------------------
// Initialize page after auth
// ------------------------
onAuthStateChanged(auth, async (user) => {
    if (!user) {
        window.location.href = "index.html";
        return;
    }

    const userUid = "tp987oF9EnnTsbNIxKdh";  // placeholder user
    const userSnap = await getDoc(doc(db, "users", userUid));
    if (userSnap.exists()) {
        const userData = userSnap.data();
        userColor = userData.color || "#0077cc";
        userName = userData.name || "Unknown User";
        if (userData.groups && userData.groups.length > 0) groupId = userData.groups[0];
    }

    const groupSnap = await getDoc(doc(db, "groups", groupId));
    if (groupSnap.exists()) {
        groupName = groupSnap.data().name || "Group";
    }

    await loadQuotes();
});