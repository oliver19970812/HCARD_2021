#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#define NOTE_B0  31
#define NOTE_C1  33
#define NOTE_CS1 35
#define NOTE_D1  37
#define NOTE_DS1 39
#define NOTE_E1  41
#define NOTE_F1  44
#define NOTE_FS1 46
#define NOTE_G1  49
#define NOTE_GS1 52
#define NOTE_A1  55
#define NOTE_AS1 58
#define NOTE_B1  62
#define NOTE_C2  65
#define NOTE_CS2 69
#define NOTE_D2  73
#define NOTE_DS2 78
#define NOTE_E2  82
#define NOTE_F2  87
#define NOTE_FS2 93
#define NOTE_G2  98
#define NOTE_GS2 104
#define NOTE_A2  110
#define NOTE_AS2 117
#define NOTE_B2  123
#define NOTE_C3  131
#define NOTE_CS3 139
#define NOTE_D3  147
#define NOTE_DS3 156
#define NOTE_E3  165
#define NOTE_F3  175
#define NOTE_FS3 185
#define NOTE_G3  196
#define NOTE_GS3 208
#define NOTE_A3  220
#define NOTE_AS3 233
#define NOTE_B3  247
#define NOTE_C4  262
#define NOTE_CS4 277
#define NOTE_D4  294
#define NOTE_DS4 311
#define NOTE_E4  330
#define NOTE_F4  349
#define NOTE_FS4 370
#define NOTE_G4  392
#define NOTE_GS4 415
#define NOTE_A4  440
#define NOTE_AS4 466
#define NOTE_B4  494
#define NOTE_C5  523
#define NOTE_CS5 554
#define NOTE_D5  587
#define NOTE_DS5 622
#define NOTE_E5  659
#define NOTE_F5  698
#define NOTE_FS5 740
#define NOTE_G5  784
#define NOTE_GS5 831
#define NOTE_A5  880
#define NOTE_AS5 932
#define NOTE_B5  988
#define NOTE_C6  1047
#define NOTE_CS6 1109
#define NOTE_D6  1175
#define NOTE_DS6 1245
#define NOTE_E6  1319
#define NOTE_F6  1397
#define NOTE_FS6 1480
#define NOTE_G6  1568
#define NOTE_GS6 1661
#define NOTE_A6  1760
#define NOTE_AS6 1865
#define NOTE_B6  1976
#define NOTE_C7  2093
#define NOTE_CS7 2217
#define NOTE_D7  2349
#define NOTE_DS7 2489
#define NOTE_E7  2637
#define NOTE_F7  2794
#define NOTE_FS7 2960
#define NOTE_G7  3136
#define NOTE_GS7 3322
#define NOTE_A7  3520
#define NOTE_AS7 3729
#define NOTE_B7  3951
#define NOTE_C8  4186
#define NOTE_CS8 4435
#define NOTE_D8  4699
#define NOTE_DS8 4978
#define REST      0

/////////////////////////////////song 1//////////////////////////////
// change this to make the song slower or faster

// change this to whichever pin you want to use
int buzzer = 8;
int melody[] = {
  NOTE_D4,4, NOTE_FS4,8, NOTE_G4,8, NOTE_A4,4, NOTE_FS4,8, NOTE_G4,8, 
  NOTE_A4,4, NOTE_B3,8, NOTE_CS4,8, NOTE_D4,8, NOTE_E4,8, NOTE_FS4,8, NOTE_G4,8, 
  NOTE_FS4,4, NOTE_D4,8, NOTE_E4,8, NOTE_FS4,4, NOTE_FS3,8, NOTE_G3,8,
  NOTE_A3,8, NOTE_G3,8, NOTE_FS3,8, NOTE_G3,8, NOTE_A3,2,
  NOTE_G3,4, NOTE_B3,8, NOTE_A3,8, NOTE_G3,4, NOTE_FS3,8, NOTE_E3,8, 
  NOTE_FS3,4, NOTE_D3,8, NOTE_E3,8, NOTE_FS3,8, NOTE_G3,8, NOTE_A3,8, NOTE_B3,8,

  NOTE_G3,4, NOTE_B3,8, NOTE_A3,8, NOTE_B3,4, NOTE_CS4,8, NOTE_D4,8,
  NOTE_A3,8, NOTE_B3,8, NOTE_CS4,8, NOTE_D4,8, NOTE_E4,8, NOTE_FS4,8, NOTE_G4,8, NOTE_A4,2,
  NOTE_A4,4, NOTE_FS4,8, NOTE_G4,8, NOTE_A4,4,
  NOTE_FS4,8, NOTE_G4,8, NOTE_A4,8, NOTE_A3,8, NOTE_B3,8, NOTE_CS4,8,
  NOTE_D4,8, NOTE_E4,8, NOTE_FS4,8, NOTE_G4,8, NOTE_FS4,4, NOTE_D4,8, NOTE_E4,8,
  NOTE_FS4,8, NOTE_CS4,8, NOTE_A3,8, NOTE_A3,8,

  NOTE_CS4,4, NOTE_B3,4, NOTE_D4,8, NOTE_CS4,8, NOTE_B3,4,
  NOTE_A3,8, NOTE_G3,8, NOTE_A3,4, NOTE_D3,8, NOTE_E3,8, NOTE_FS3,8, NOTE_G3,8,
  NOTE_A3,8, NOTE_B3,4, NOTE_G3,4, NOTE_B3,8, NOTE_A3,8, NOTE_B3,4,
  NOTE_CS4,8, NOTE_D4,8, NOTE_A3,8, NOTE_B3,8, NOTE_CS4,8, NOTE_D4,8, NOTE_E4,8,
  NOTE_FS4,8, NOTE_G4,8, NOTE_A4,2,  
   
  
};
// sizeof gives the number of bytes, each int value is composed of two bytes (16 bits)
// there are two values per note (pitch and duration), so for each note there are four bytes
int notes = sizeof(melody) / sizeof(melody[0]) / 2;
// this calculates the duration of a whole note in ms
int tempo = 100;
int wholenote = (60000 * 4) / tempo;
int divider = 0, noteDuration = 0;
//int thisNote = 0;
//////////////////////song 2////////////////////////////////////
int Pink[] = {
  REST,2, REST,4, REST,8, NOTE_DS4,8, 
  NOTE_E4,-4, REST,8, NOTE_FS4,8, NOTE_G4,-4, REST,8, NOTE_DS4,8,
  NOTE_E4,-8, NOTE_FS4,8,  NOTE_G4,-8, NOTE_C5,8, NOTE_B4,-8, NOTE_E4,8, NOTE_G4,-8, NOTE_B4,8,   
  NOTE_AS4,2, NOTE_A4,-16, NOTE_G4,-16, NOTE_E4,-16, NOTE_D4,-16, 
  NOTE_E4,2, REST,4, REST,8, NOTE_DS4,4,

  NOTE_E4,-4, REST,8, NOTE_FS4,8, NOTE_G4,-4, REST,8, NOTE_DS4,8,
  NOTE_E4,-8, NOTE_FS4,8,  NOTE_G4,-8, NOTE_C5,8, NOTE_B4,-8, NOTE_G4,8, NOTE_B4,-8, NOTE_E5,8,
  NOTE_DS5,1,   
  NOTE_D5,2, REST,4, REST,8, NOTE_DS4,8, 
  NOTE_E4,-4, REST,8, NOTE_FS4,8, NOTE_G4,-4, REST,8, NOTE_DS4,8,
  NOTE_E4,-8, NOTE_FS4,8,  NOTE_G4,-8, NOTE_C5,8, NOTE_B4,-8, NOTE_E4,8, NOTE_G4,-8, NOTE_B4,8,   
  
  NOTE_AS4,2, NOTE_A4,-16, NOTE_G4,-16, NOTE_E4,-16, NOTE_D4,-16, 
  NOTE_E4,-4, REST,4,
  REST,4, NOTE_E5,-8, NOTE_D5,8, NOTE_B4,-8, NOTE_A4,8, NOTE_G4,-8, NOTE_E4,-8,
  NOTE_AS4,16, NOTE_A4,-8, NOTE_AS4,16, NOTE_A4,-8, NOTE_AS4,16, NOTE_A4,-8, NOTE_AS4,16, NOTE_A4,-8,   
  NOTE_G4,-16, NOTE_E4,-16, NOTE_D4,-16, NOTE_E4,16, NOTE_E4,16, NOTE_E4,2,
};
int Pink_tempo=120;
int Pink_notes = sizeof(Pink) / sizeof(Pink[0]) / 2;
// this calculates the duration of a whole note in ms
int Pink_wholenote = (60000 * 4) / Pink_tempo;
int Pink_divider = 0, Pink_noteDuration = 0;

int Star[] = {
  
  REST, 2, NOTE_D4, 4,
  NOTE_G4, -4, NOTE_AS4, 8, NOTE_A4, 4,
  NOTE_G4, 2, NOTE_D5, 4,
  NOTE_C5, -2, 
  NOTE_A4, -2,
  NOTE_G4, -4, NOTE_AS4, 8, NOTE_A4, 4,
  NOTE_F4, 2, NOTE_GS4, 4,
  NOTE_D4, -1, 
  NOTE_D4, 4,

  NOTE_G4, -4, NOTE_AS4, 8, NOTE_A4, 4, //10
  NOTE_G4, 2, NOTE_D5, 4,
  NOTE_F5, 2, NOTE_E5, 4,
  NOTE_DS5, 2, NOTE_B4, 4,
  NOTE_DS5, -4, NOTE_D5, 8, NOTE_CS5, 4,
  NOTE_CS4, 2, NOTE_B4, 4,
  NOTE_G4, -1,
  NOTE_AS4, 4,
     
  NOTE_D5, 2, NOTE_AS4, 4,//18
  NOTE_D5, 2, NOTE_AS4, 4,
  NOTE_DS5, 2, NOTE_D5, 4,
  NOTE_CS5, 2, NOTE_A4, 4,
  NOTE_AS4, -4, NOTE_D5, 8, NOTE_CS5, 4,
  NOTE_CS4, 2, NOTE_D4, 4,
  NOTE_D5, -1, 
  REST,4, NOTE_AS4,4,  

  NOTE_D5, 2, NOTE_AS4, 4,//26
  NOTE_D5, 2, NOTE_AS4, 4,
  NOTE_F5, 2, NOTE_E5, 4,
  NOTE_DS5, 2, NOTE_B4, 4,
  NOTE_DS5, -4, NOTE_D5, 8, NOTE_CS5, 4,
  NOTE_CS4, 2, NOTE_AS4, 4,
  NOTE_G4, -1, 
  
};
int Star_tempo=140;
int Star_notes = sizeof( Star) / sizeof( Star[0]) / 2;
// this calculates the duration of a whole note in ms
int  Star_wholenote = (60000 * 4) /  Star_tempo;
int  Star_divider = 0,  Star_noteDuration = 0;


#define BLE_UUID_PERIPHERAL               "19B10000-E8F2-537E-4F6C-D104768A1214"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_LED              "19B10001-E8F2-537E-4F6C-E104768A1214"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_ACCX             "29B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_ACCY             "39B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_ACCZ             "49B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
//#define BLE_UUID_CHARACT_tem             "59B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
// ------------------------------------------ VOID SETUP ------------------------------------------
void setup() {
  Serial.begin(9600);
  while (!Serial);
  BLE.begin();
  BLE.scanForUuid(BLE_UUID_PERIPHERAL);//
 
}

byte DataToRead [6];

// ------------------------------------------ VOID LOOP ------------------------------------------
void loop() {
  // check if a peripheral has been discovered
  BLEDevice peripheral = BLE.available();
 // int datafromUser=0;
 //  if (Serial.available()>0)
 // {datafromUser=(int)Serial.read();  
   // datafromUser=Serial.read();
  //datafromUser=int(datafromUser);
//  Serial.print("user number is ");
//  Serial.print(datafromUser);
// }
  
  if (peripheral) {
     if (peripheral.localName() != "BLE_IMU") {
      return;
    }
    // stop scanning
    BLE.stopScan();
    LED_IMU(peripheral);
    // peripheral disconnected, start scanning again
    BLE.scanForUuid(BLE_UUID_PERIPHERAL);}
}

// ------------------------------------------ FUNCTIONS ------------------------------------------
void LED_IMU(BLEDevice peripheral) {
  int flag=0;
  if(Serial.available()>0)
 {flag=(int)Serial.read();  
  //Serial.print("flag is ");
  //Serial.print(flag);
 
  if (peripheral.connect()) {
  } else {
    return;
  }
 }  //while

//  Serial.println("Discovering attributes ...");
  if (peripheral.discoverAttributes()) {
//    Serial.println("Attributes discovered");
  } 
  else {
//    Serial.println("Attribute discovery failed!");
    peripheral.disconnect();
    return;
  }

  // retrieve the LED characteristic
 
  BLECharacteristic accXCharacteristic = peripheral.characteristic(BLE_UUID_CHARACT_ACCX);
  BLECharacteristic accYCharacteristic = peripheral.characteristic(BLE_UUID_CHARACT_ACCY);
  BLECharacteristic accZCharacteristic = peripheral.characteristic(BLE_UUID_CHARACT_ACCZ);
 
  
  float x, y, z;
  while (peripheral.connected()) {
    
    if(flag == 48){
    for ( int thisNote=0 ; thisNote < notes * 2; thisNote = thisNote + 2) {
    // calculates the duration of each note
    divider = melody[thisNote + 1];
    if (divider > 0) {
      // regular note, just proceed
      noteDuration = (wholenote) / divider;
    } else if (divider < 0) {
      // dotted notes are represented with negative durations!!
      noteDuration = (wholenote) / abs(divider);
      noteDuration *= 1.5; // increases the duration in half for dotted notes
    }

    // we only play the note for 90% of the duration, leaving 10% as a pause
    tone(buzzer, melody[thisNote], noteDuration * 0.9);
    // Wait for the specief duration before playing the next note.
    delay(noteDuration);
    // stop the waveform generation before the next note.
    noTone(buzzer);
    accXCharacteristic.readValue( &x, 4 );
    accYCharacteristic.readValue( &y, 4 );
    accZCharacteristic.readValue( &z, 4 );
   //  temCharacteristic.readValue( &t,4);
    Serial.print(x);
    Serial.print(' ');
    Serial.print(y);
    Serial.print(' ');       
    Serial.print(z);
    Serial.print('\n');
    if ((abs(x)<5&&abs(y)<5&&abs(z)<5)||((abs(x)>160)||(abs(y)>160)||(abs(z)>160))){
    //digitalWrite (buzzer,HIGH );
    //delay (500);
    digitalWrite (buzzer, LOW);
    delay (600);}
   }//for
  }//if 
else if(flag == 49)
{
  for ( int Pink_thisNote=0 ; Pink_thisNote < Pink_notes * 2; Pink_thisNote = Pink_thisNote + 2) {
         Pink_divider = Pink[Pink_thisNote + 1];
    if (Pink_divider > 0) {
     
      Pink_noteDuration = (Pink_wholenote) / Pink_divider;
    } else if (Pink_divider < 0) {
      Pink_noteDuration = (Pink_wholenote) / abs(Pink_divider);
      Pink_noteDuration *= 1.5; // increases the duration in half for dotted notes
    }
    tone(buzzer, Pink[Pink_thisNote], Pink_noteDuration * 0.9);
    delay(Pink_noteDuration);
    noTone(buzzer);
    accXCharacteristic.readValue( &x, 4 );
    accYCharacteristic.readValue( &y, 4 );
    accZCharacteristic.readValue( &z, 4 );
    Serial.print(x);
    Serial.print(' ');
    Serial.print(y);
    Serial.print(' ');       
    Serial.print(z);
    Serial.print('\n');

  if ((abs(x)<5&&abs(y)<5&&abs(z)<5)||((abs(x)>160)||(abs(y)>160)||(abs(z)>160))){
  //  digitalWrite (buzzer,HIGH );
  //  delay (500);
    digitalWrite (buzzer, LOW);
    delay (600);}
  }//for
}//else if
else{
  for ( int Star_thisNote=0 ; Star_thisNote < Star_notes * 2; Star_thisNote = Star_thisNote + 2) {
         Star_divider = Star[Star_thisNote + 1];
    if (Star_divider > 0) {
     
      Star_noteDuration = (Star_wholenote) / Star_divider;
    } else if (Star_divider < 0) {
      Star_noteDuration = (Star_wholenote) / abs(Star_divider);
      Star_noteDuration *= 1.5; // increases the duration in half for dotted notes
    }
    tone(buzzer, Star[Star_thisNote], Star_noteDuration * 0.9);
    delay(Star_noteDuration);
    noTone(buzzer);
    accXCharacteristic.readValue( &x, 4 );
    accYCharacteristic.readValue( &y, 4 );
    accZCharacteristic.readValue( &z, 4 );
    Serial.print(x);
    Serial.print(' ');
    Serial.print(y);
    Serial.print(' ');       
    Serial.print(z);
    Serial.print('\n');

  if ((abs(x)<5&&abs(y)<5&&abs(z)<5)||((abs(x)>160)||(abs(y)>160)||(abs(z)>160))){
 //   digitalWrite (buzzer,HIGH );
 //   delay (500);
    digitalWrite (buzzer, LOW);
    delay (600);}
  }//for
  }//else
  }//while
//  Serial.println("Peripheral disconnected");
}
