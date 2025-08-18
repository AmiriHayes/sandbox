import * as functions from "firebase-functions";
import express, { Request, Response } from "express";
import bodyParser from "body-parser";
import cors from "cors";
import { db } from "./firebase";

const app = express();
app.use(bodyParser.json());
app.use(cors());

interface Book {
    id?: number;
    title: string;
    author: string;
}

// Helper to get the next ID (optional)
const getNextId = async (): Promise<number> => {
    const snapshot = await db.collection("books").orderBy("id", "desc").limit(1).get();
    if (snapshot.empty) return 1;
    return snapshot.docs[0].data().id + 1;
};

app.get("/", (req: Request, res: Response) => {
    res.send("Welcome to the Books API with Firestore!");
});

app.get("/books", async (req: Request, res: Response) => {
    const snapshot = await db.collection("books").get();
    const books: Book[] = snapshot.docs.map(doc => doc.data() as Book);
    res.json(books);
});

app.get("/books/:id", async (req: Request, res: Response) => {
    const id = parseInt(req.params.id);
    const snapshot = await db.collection("books").where("id", "==", id).get();

    if (snapshot.empty) {
        return res.status(404).json({ message: "Book not found" });
    }

    res.json(snapshot.docs[0].data());
});

app.post("/books", async (req: Request, res: Response) => {
    const { title, author } = req.body;

    if (!title || !author) {
        return res.status(400).json({ message: "Title and author are required." });
    }

    const id = await getNextId();
    const newBook: Book = { id, title, author };

    await db.collection("books").add(newBook);
    res.status(201).json(newBook);
});

app.put("/books/:id", async (req: Request, res: Response) => {
    const id = parseInt(req.params.id);
    const { title, author } = req.body;

    const snapshot = await db.collection("books").where("id", "==", id).get();
    if (snapshot.empty) return res.status(404).json({ message: "Book not found" });

    const docId = snapshot.docs[0].id;
    await db.collection("books").doc(docId).set({ id, title, author });

    res.json({ id, title, author });
});

app.patch("/books/:id", async (req: Request, res: Response) => {
    const id = parseInt(req.params.id);
    const { title, author } = req.body;

    const snapshot = await db.collection("books").where("id", "==", id).get();
    if (snapshot.empty) return res.status(404).json({ message: "Book not found" });

    const docId = snapshot.docs[0].id;
    const currentData = snapshot.docs[0].data();

    const updatedBook: Book = {
        id,
        title: title ?? currentData.title,
        author: author ?? currentData.author,
    };

    await db.collection("books").doc(docId).set(updatedBook);
    res.json(updatedBook);
});

app.delete("/books/:id", async (req: Request, res: Response) => {
    const id = parseInt(req.params.id);
    const snapshot = await db.collection("books").where("id", "==", id).get();

    if (snapshot.empty) return res.status(404).json({ message: "Book not found" });

    const docId = snapshot.docs[0].id;
    const deletedBook = snapshot.docs[0].data();
    await db.collection("books").doc(docId).delete();

    res.json({ message: "Book deleted", book: deletedBook });
});

// Export as Firebase Function
export const api = functions.https.onRequest(app);
