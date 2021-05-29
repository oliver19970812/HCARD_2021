#include <ArduinoBLE.h>
//#include <Arduino_LSM9DS1.h>
#define BLE_UUID_PERIPHERAL               "19B10000-E8F2-537E-4F6C-D104768A1214"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_LED              "19B10001-E8F2-537E-4F6C-E104768A1214"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_ACCX             "29B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_ACCY             "39B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL
#define BLE_UUID_CHARACT_ACCZ             "49B10001-E8F2-537E-4F6C-a204768A1215"  //please change to a unique value that matches BLE_IMU_PERIPHERAL

// ------------------------------------------ VOID SETUP ------------------------------------------
void setup() {
  Serial.begin(9600);
  while (!Serial);
  BLE.begin();
  BLE.scanForUuid(BLE_UUID_PERIPHERAL);//
 
}
// ------------------------------------------ VOID LOOP ------------------------------------------
void loop() {
  // check if a peripheral has been discovered
  BLEDevice peripheral = BLE.available();

  
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
  if (peripheral.connect()) {
  } else {
    return;
  }
  //while

//  Serial.println("Discovering attributes ...");
  if (peripheral.discoverAttributes()){
//    Serial.println("Attributes discovered");
  } 
  else {
//    Serial.println("Attribute discovery failed!");
    peripheral.disconnect();
    return;
  }
  BLECharacteristic accXCharacteristic = peripheral.characteristic(BLE_UUID_CHARACT_ACCX);
  BLECharacteristic accYCharacteristic = peripheral.characteristic(BLE_UUID_CHARACT_ACCY);
  BLECharacteristic accZCharacteristic = peripheral.characteristic(BLE_UUID_CHARACT_ACCZ);
  float x, y, z;
  while (peripheral.connected()) {
    accXCharacteristic.readValue( &x, 4 );
    accYCharacteristic.readValue( &y, 4 );
    accZCharacteristic.readValue( &z, 4 );
    Serial.print(x);
    Serial.print(' ');
    Serial.print(y);
    Serial.print(' ');       
    Serial.print(z);
    Serial.print('\n');
    delay(800);
  }//while
//  Serial.println("Peripheral disconnected");
}
