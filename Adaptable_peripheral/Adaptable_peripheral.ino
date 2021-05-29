/*
  -----------------------------------------------------------------------------------------------
  | BLE_IMU_PERIPHERAL - Wireless IMU Communication with central device
  |
  | Arduino Boards Tested: Nano 33 BLE Sense as a peripheral & Nano 33 BLE as central.
  | Code not tested for multiple peripherals

  | This sketch works alongside the BLE_IMU_CENTRAL sketch to communicate with an Arduino Nano 33 BLE.
  | This sketch can also be used with a generic BLE central app, like LightBlue (iOS and Android) or
  | nRF Connect (Android), to interact with the services and characteristics created in this sketch.

  | This example code is adapted from the ArduinoBLE library, available in the public domain.
  | Authors: Aaron Yurkewich & Pilar Zhang Qiu
  | Latest Update: 25/02/2021
  -----------------------------------------------------------------------------------------------
*/
#include <ArduinoBLE.h>
#include <Arduino_LSM9DS1.h>
#include <Arduino_HTS221.h>
#include "Arduino.h"
#define BLUE 11
#define GREEN 10
#define RED 9
int redValue = 100;
int greenValue = 100;
int blueValue = 100;
float temp_cali;
float temperature;
float humidity;
const int ledPin = LED_BUILTIN; // pin to use for the LED
float x, y, z;
int counter = 0;
// ------------------------------------------ BLE UUIDs ------------------------------------------
#define BLE_UUID_PERIPHERAL               "19B10000-E8F2-537E-4F6C-D104768A1214" //please chnage to a unique value that matches BLE_IMU_CENTRAL
#define BLE_UUID_CHARACT_LED              "19B10001-E8F2-537E-4F6C-E104768A1214" //please chnage to a unique value that matches BLE_IMU_CENTRAL
#define BLE_UUID_CHARACT_ACCX             "29B10001-E8F2-537E-4F6C-a204768A1215" //please chnage to a unique value that matches BLE_IMU_CENTRAL
#define BLE_UUID_CHARACT_ACCY             "39B10001-E8F2-537E-4F6C-a204768A1215" //please chnage to a unique value that matches BLE_IMU_CENTRAL
#define BLE_UUID_CHARACT_ACCZ             "49B10001-E8F2-537E-4F6C-a204768A1215" //please chnage to a unique value that matches BLE_IMU_CENTRAL
//#define BLE_UUID_CHARACT_tem             "59B10001-E8F2-537E-4F6C-a204768A1215" //please chnage to a unique value that matches BLE_IMU_CENTRAL

BLEService LED_IMU_Service(BLE_UUID_PERIPHERAL); // BLE LED Service

// BLE LED Switch Characteristic - custom 128-bit UUID, read and writable by central
BLEByteCharacteristic switchCharacteristic(BLE_UUID_CHARACT_LED, BLERead | BLEWrite);
BLEFloatCharacteristic accXCharacteristic(BLE_UUID_CHARACT_ACCX, BLERead | BLENotify | BLEWrite);
BLEFloatCharacteristic accYCharacteristic(BLE_UUID_CHARACT_ACCY, BLERead | BLENotify | BLEWrite);
BLEFloatCharacteristic accZCharacteristic(BLE_UUID_CHARACT_ACCZ, BLERead | BLENotify | BLEWrite);
//BLEFloatCharacteristic temCharacteristic(BLE_UUID_CHARACT_tem, BLERead | BLENotify | BLEWrite);





// ------------------------------------------ VOID SETUP ------------------------------------------
void setup() {
  Serial.begin(9600);
  //while (!Serial); //uncomment to view the IMU data in the peripheral serial monitor

  // begin IMU initialization
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }


  // begin BLE initialization
  if (!BLE.begin()) {
    Serial.println("starting BLE failed!");
    while (1);

  }

  // set advertised local name and service UUID:
  BLE.setLocalName("BLE_IMU");
  BLE.setAdvertisedService(LED_IMU_Service);

  // add the characteristic to the service
  LED_IMU_Service.addCharacteristic(switchCharacteristic);
  LED_IMU_Service.addCharacteristic(accXCharacteristic);
  LED_IMU_Service.addCharacteristic(accYCharacteristic);
  LED_IMU_Service.addCharacteristic(accZCharacteristic);
  //LED_IMU_Service.addCharacteristic(temCharacteristic);

  // add service
  BLE.addService(LED_IMU_Service);

  // set the initial value for the characeristic:
  switchCharacteristic.writeValue(0);


  // start advertising
  BLE.advertise();

  Serial.println("BLE LED Peripheral");

  if (!HTS.begin()) {
    Serial.println("Failed to initialize humidity temperature sensor!");
    while (1);
  }
  pinMode(RED, OUTPUT);
  pinMode(GREEN, OUTPUT);
  pinMode(BLUE, OUTPUT);
  digitalWrite(RED, HIGH);
  digitalWrite(GREEN, LOW);
  digitalWrite(BLUE, LOW);
  analogWrite(BLUE, blueValue);
  analogWrite(RED, redValue);
  analogWrite(GREEN, greenValue);
}

// ------------------------------------------ VOID LOOP ------------------------------------------
void loop() {
  // listen for BLE peripherals to connect:
  BLEDevice central = BLE.central();
  
  // if a central is connected to peripheral:
  if (central) {
    Serial.print("Connected to central: ");
    // print the central's MAC address:
    Serial.println(central.address());
  }
// while the central is still connected to peripheral:
 while (central.connected()) {

 //  if (HTS.begin()) {
//    temperature = HTS.readTemperature();
//    temCharacteristic.writeValue(temperature);
//    Serial.print("Temperature = ");
//    Serial.print(temperature);
//    Serial.println(" °C");
//    Serial.print('\t');
//    HTS.end();

//  }
 
  
  // print an empty line
  Serial.println();

  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(x, y, z);

    accXCharacteristic.writeValue(x);
    accYCharacteristic.writeValue(y);
    accZCharacteristic.writeValue(z);
    Serial.print(x);
    Serial.print('\t');
    Serial.print(y);
    Serial.print('\t');
    Serial.print(z);
    Serial.println('\t');
  }
  Serial.print ("test1");
    }
Serial.print ("we have left the while loop ");
Serial.println('\t');
}
