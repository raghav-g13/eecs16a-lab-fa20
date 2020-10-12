// defining the pin mapping
#define Ypos A0
#define Xpos A1
#define Xneg A2
#define Yneg A3

// global variables
uint32_t x_raw, y_raw;
uint8_t coordinates[] = {0, 0};

/*  ToDo: Finish this function
    getLoc returns integer value coordinates of
    where a touch was detected
*/
void getLoc(uint32_t x_raw, uint32_t y_raw) {
  float raw_range = 4096.0;
  int width  = 3;
  int height = 3;

  // BEGIN {{
  int x_coordinate = // YOUR CODE HERE
  int y_coordinate = // YOUR CODE HERE
  // }} END

  coordinates[0] = (uint8_t) x_coordinate;
  coordinates[1] = (uint8_t) y_coordinate;
}

void setup()
{
  // start serial port at 115200 bps:
  Serial.begin(115200);
  while (!Serial);
  Serial.println("+-----------------------+");
  Serial.println("| Resistive Touchscreen |");
  Serial.println("| Start Pressing Points |"); 
  Serial.println("+-----------------------+");
}

void loop() {
  if (touch()) {
    x_raw = meas(Xneg, Xpos, Yneg, Ypos);
    y_raw = meas(Yneg, Ypos, Xneg, Xpos);
    getLoc(x_raw, y_raw);
    Serial.print("X = ");
    Serial.print(coordinates[0]);
    Serial.print(" Y = ");
    Serial.println(coordinates[1]);
    delay(1000);
  }
}

/* touch returns true if a touch has been detected;
  returns false otherwise.  */
boolean touch()
{
  pinMode(Ypos, INPUT_PULLUP);
  pinMode(Yneg, INPUT);
  pinMode(Xneg, OUTPUT);
  pinMode(Xpos, INPUT);
  digitalWrite(Xneg, LOW);
  boolean touch = false;
  if (!digitalRead(Ypos)) {
    touch = true;
  }
  //Serial.println(touch);
  return touch;
}

uint32_t meas(int pwr_neg, int pwr_pos, int sense_neg, int sense_pos) {
  pinMode(sense_neg, INPUT); // sets sense_neg to be an input pin
  pinMode(sense_pos, INPUT); // sets sense_pos to be an input pin
  digitalWrite(sense_neg, LOW); // outputs GND to sense_neg to protect from floating voltages
  pinMode(pwr_neg, OUTPUT); // sets pwr_neg to output
  digitalWrite(pwr_neg, LOW); // outputs GND to pwr_neg
  pinMode(pwr_pos, OUTPUT); // sets pwr_pos to output
  digitalWrite(pwr_pos, HIGH); // outputs +3V to pwr_pos
  return analogRead(sense_pos);
}
