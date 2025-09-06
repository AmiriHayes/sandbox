// ------------------- Imports -------------------
import express, { Request, Response } from "express";
import bodyParser from "body-parser";

// ------------------- Setup app -------------------
const app = express();
const PORT = 3000;

app.use(bodyParser.json());

// ------------------- Types -------------------
interface Book {
  id: number;
  title: string;
  author: string;
}

// ------------------- Mock "database" -------------------
let books: Book[] = [
  { id: 1, title: "1984", author: "George Orwell" },
  { id: 2, title: "Brave New World", author: "Aldous Huxley" },
];

// ------------------- Routes -------------------

// Root route
app.get("/", (req: Request, res: Response): void => {
  res.send("Welcome to the Books API (TypeScript version)!");
});

// ------------------- READ all -------------------
app.get("/books", (req: Request, res: Response): void => {
  res.json(books);
});

// ------------------- READ one -------------------
app.get("/books/:id", (req: Request, res: Response): void => {
  const id = parseInt(req.params.id);
  const book = books.find((b) => b.id === id);

  if (book) {
    res.json(book);
  } else {
    res.status(404).json({ message: "Book not found" });
  }
});

// ------------------- CREATE -------------------
app.post("/books", (req: Request, res: Response): void => {
  const { title, author } = req.body;

  if (!title || !author) {
    res.status(400).json({ message: "Title and author are required" });
    return;
  }

  const newBook: Book = {
    id: books.length ? books[books.length - 1].id + 1 : 1,
    title,
    author,
  };

  books.push(newBook);
  res.status(201).json(newBook);
});

// ------------------- UPDATE -------------------
app.put("/books/:id", (req: Request, res: Response): void => {
  const id = parseInt(req.params.id);
  const { title, author } = req.body;
  const index = books.findIndex((b) => b.id === id);

  if (index !== -1) {
    books[index] = { id, title, author };
    res.json(books[index]);
  } else {
    res.status(404).json({ message: "Book not found" });
  }
});

// ------------------- PATCH -------------------
app.patch("/books/:id", (req: Request, res: Response): void => {
  const id = parseInt(req.params.id);
  const { title, author } = req.body;
  const book = books.find((b) => b.id === id);

  if (book) {
    if (title) book.title = title;
    if (author) book.author = author;
    res.json(book);
  } else {
    res.status(404).json({ message: "Book not found" });
  }
});

// ------------------- DELETE -------------------
app.delete("/books/:id", (req: Request, res: Response): void => {
  const id = parseInt(req.params.id);
  const index = books.findIndex((b) => b.id === id);

  if (index !== -1) {
    const deletedBook = books.splice(index, 1)[0];
    res.json({ message: "Book deleted", book: deletedBook });
  } else {
    res.status(404).json({ message: "Book not found" });
  }
});

// ------------------- Start server -------------------
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
