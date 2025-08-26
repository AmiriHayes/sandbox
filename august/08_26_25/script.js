class EMNISTCanvas {
            constructor() {
                this.canvas = document.getElementById('drawingCanvas');
                this.ctx = this.canvas.getContext('2d');
                this.isDrawing = false;
                this.setupCanvas();
                this.bindEvents();
            }

            setupCanvas() {
                // Set up high-resolution canvas
                const rect = this.canvas.getBoundingClientRect();
                const scale = window.devicePixelRatio || 1;
                
                this.canvas.width = 28;
                this.canvas.height = 28;
                
                // Fill with white background
                this.ctx.fillStyle = 'white';
                this.ctx.fillRect(0, 0, 28, 28);
                
                // Set drawing properties
                this.ctx.strokeStyle = 'black';
                this.ctx.lineWidth = 2;
                this.ctx.lineCap = 'round';
                this.ctx.lineJoin = 'round';
            }

            bindEvents() {
                // Mouse events
                this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
                this.canvas.addEventListener('mousemove', (e) => this.draw(e));
                this.canvas.addEventListener('mouseup', () => this.stopDrawing());
                this.canvas.addEventListener('mouseout', () => this.stopDrawing());

                // Touch events for mobile
                this.canvas.addEventListener('touchstart', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const mouseEvent = new MouseEvent('mousedown', {
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    });
                    this.canvas.dispatchEvent(mouseEvent);
                });

                this.canvas.addEventListener('touchmove', (e) => {
                    e.preventDefault();
                    const touch = e.touches[0];
                    const mouseEvent = new MouseEvent('mousemove', {
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    });
                    this.canvas.dispatchEvent(mouseEvent);
                });

                this.canvas.addEventListener('touchend', (e) => {
                    e.preventDefault();
                    const mouseEvent = new MouseEvent('mouseup', {});
                    this.canvas.dispatchEvent(mouseEvent);
                });
            }

            getCoordinates(event) {
                const rect = this.canvas.getBoundingClientRect();
                const scaleX = this.canvas.width / rect.width;
                const scaleY = this.canvas.height / rect.height;

                const clientX = event.touches ? event.touches[0].clientX : event.clientX;
                const clientY = event.touches ? event.touches[0].clientY : event.clientY;

                return {
                    x: (clientX - rect.left) * scaleX,
                    y: (clientY - rect.top) * scaleY
                };
            }

            startDrawing(event) {
                this.isDrawing = true;
                const coords = this.getCoordinates(event);
                this.ctx.beginPath();
                this.ctx.moveTo(coords.x, coords.y);
            }

            draw(event) {
                if (!this.isDrawing) return;
                const coords = this.getCoordinates(event);
                this.ctx.lineTo(coords.x, coords.y);
                this.ctx.stroke();
            }

            stopDrawing() {
                if (this.isDrawing) {
                    this.isDrawing = false;
                    this.ctx.beginPath();
                }
            }

            clear() {
                this.ctx.fillStyle = 'white';
                this.ctx.fillRect(0, 0, 28, 28);
                this.hideResults();
            }

            getImageData() {
                // Get image data and convert to grayscale array
                const imageData = this.ctx.getImageData(0, 0, 28, 28);
                const pixels = imageData.data;
                const grayscale = [];

                for (let i = 0; i < pixels.length; i += 4) {
                    const gray = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3;
                    grayscale.push((255 - gray) / 255);
                }

                return grayscale;
            }
            hideResults() {
                document.getElementById('results').style.display = 'none';
            }
        }

        class PredictionAPI {
            constructor() {
                this.apiUrl = 'https://amirihayes-alphanumeric-ml.hf.space/predict';
            }
            async predict(imageData) {
                const response = await fetch(this.apiUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        input_image: imageData
                    })
                });
                if (!response.ok) {
                    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
                }
                return await response.json();
            }
        }

        class App {
            constructor() {
                this.canvas = new EMNISTCanvas();
                this.api = new PredictionAPI();
                this.bindControls();
            }

            bindControls() {
                document.getElementById('clearBtn').addEventListener('click', () => {
                    this.canvas.clear();
                });
                document.getElementById('submitBtn').addEventListener('click', () => {
                    this.handleSubmit();
                });
            }

            async handleSubmit() {
                const submitBtn = document.getElementById('submitBtn');
                const loading = document.getElementById('loading');
                const error = document.getElementById('error');
                const results = document.getElementById('results');

                try {
                    // Show loading state
                    submitBtn.disabled = true;
                    loading.style.display = 'block';
                    error.style.display = 'none';
                    results.style.display = 'none';

                    // Get image data
                    const imageData = this.canvas.getImageData();
                    
                    // Make prediction
                    const result = await this.api.predict(imageData);
                    
                    // Display results
                    this.displayResults(result);

                } catch (err) {
                    console.error('Prediction error:', err);
                    error.textContent = `Error: ${err.message}. Make sure the API server is running on localhost:8000`;
                    error.style.display = 'block';
                } finally {
                    // Hide loading state
                    submitBtn.disabled = false;
                    loading.style.display = 'none';
                }
            }

            displayResults(result) {
                const charElement = document.getElementById('predictedChar');
                const confidenceElement = document.getElementById('confidencePercent');
                const results = document.getElementById('results');

                // Update display
                charElement.textContent = result.prediction;
                confidenceElement.textContent = `${Math.round(result.confidence * 100)}%`;

                // Show results
                results.style.display = 'flex';

                // Smooth scroll to results
                results.scrollIntoView({ 
                    behavior: 'smooth',
                    block: 'center'
                });
            }
        }

        document.addEventListener('DOMContentLoaded', () => {
            new App();
        });