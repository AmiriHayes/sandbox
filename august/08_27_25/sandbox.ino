const int buttonPin = 2;
int buttonState = 0;
int lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;

void setup() {
  Serial.begin(9600);
  pinMode(buttonPin, INPUT_PULLUP);
  
  Serial.println("Button Press Logger Started");
  Serial.println("Press the button to log timestamps...");
  Serial.println("---");
}

void loop() {
  int reading = digitalRead(buttonPin);
  
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      
      if (buttonState == LOW) {
        unsigned long timestamp = millis();
        
        Serial.print("Button pushed @ ");
        Serial.print(timestamp);
        Serial.println(" ms");
      }
    }
  }

  lastButtonState = reading;
}