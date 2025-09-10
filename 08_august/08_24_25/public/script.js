import { initializeApp } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-app.js";
import { getAuth, GoogleAuthProvider, signInWithPopup, onAuthStateChanged } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-auth.js";
import { getFirestore, doc, getDoc, setDoc } from "https://www.gstatic.com/firebasejs/12.1.0/firebase-firestore.js";

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
const provider = new GoogleAuthProvider();

const colorSection = document.getElementById("colorSection");
const userColorInput = document.getElementById("userColor");
const signInButton = document.getElementById("googleSignIn");

// async function handleSignIn() {
//     try {
//         const result = await signInWithPopup(auth, provider);
//         const user = result.user;
//         if (!user) return;

//         const userRef = doc(db, "users", user.uid);
//         const userSnap = await getDoc(userRef);

//         if (!userSnap.exists()) {
//             // Show color section
//             colorSection.style.display = "block";

//             // Wait for user to click "Confirm"
//             await new Promise(resolve => {
//                 document.getElementById("confirmColor").addEventListener("click", () => {
//                     resolve(userColorInput.value);
//                 }, { once: true });
//             }).then(async (selectedColor) => {
//                 // Create user in Firestore
//                 await setDoc(userRef, {
//                     name: user.displayName || "Unknown",
//                     email: user.email,
//                     color: selectedColor,
//                     groups: []
//                 });

//                 // Redirect to member.html
//                 window.location.href = "member.html";
//             });
//         } else {
//             // Existing user: redirect immediately
//             window.location.href = "member.html";
//         }

//     } catch (error) {
//         console.error("Error signing in:", error);
//     }
// }

async function handleSignIn() {
    try {
        const result = await signInWithPopup(auth, provider);
        const user = result.user;
        if (!user) return;

        const userRef = doc(db, "users", user.uid);
        const userSnap = await getDoc(userRef);

        if (userSnap.exists()) {
            // Existing user: redirect immediately
            window.location.href = "member.html";
        } else {
            // New user: show color picker
            colorSection.style.display = "block";

            const selectedColor = await new Promise(resolve => {
                document.getElementById("confirmColor").addEventListener("click", () => {
                    resolve(userColorInput.value);
                }, { once: true });
            });

            // Create new user in Firestore
            await setDoc(userRef, {
                name: user.displayName || "Unknown",
                email: user.email,
                color: selectedColor,
                groups: []
            });

            // Redirect after creation
            window.location.href = "member.html";
        }
    } catch (error) {
        console.error("Error signing in:", error);
    }
}


signInButton.addEventListener("click", handleSignIn);

// Optional: redirect if already signed in
// onAuthStateChanged(auth, (user) => {
//     if (user) window.location.href = "member.html";
// });
