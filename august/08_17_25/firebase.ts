import admin from "firebase-admin";

admin.initializeApp(); // Uses credentials from your Firebase project automatically

export const db = admin.firestore();
