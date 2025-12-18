# FastAPI farcaster frames server for tumbller rover


## Setup

* To run the robots first thing to do is the wifi setup. This computer running this server and both the ESP32s need to be on the same network or VPN. The following instructions will set it up for the devices on the same network. 
* Clone the follwoing three repos in a folder
  * https://github.com/YakRoboticsGarage/tumbller-fc-frames-fastapi
  * https://github.com/YakRoboticsGarage/tumbller-esp32s3
  * https://github.com/YakRoboticsGarage/tumbller-esp-cam
  
* Both tumbller-esp32s3 and tumbller-esp-cam have WIFI_SSID and WIFI_PASSWORD hardcoded. Find them and replace with your own wifi credentials
* Now we need to know the IPs of each of the ESPs and the computer this server will run on.
* The ESP-CAM will spit out the IP on the serial port when connected to a serial terminal. You should see something like this on the serial terminal when you press the __RESET__ button (the right button when the camera is facing you). The ESP-CAM will fast blink in red color when it is trying to connect to WiFi and will blink in White when connected. 
  * Note down the ESP-CAM-IP

```

WiFi connected
Camera Ready! Use 'http://ESP-CAM-IP/stream' to connect
HTTP server started
Camera sensor verified and ready
Signal strength (RSSI): -45 dBm
TX Power: 78 dBm

```

* For tumbller-esp32s3 we will need to enable the serial output in the code to find the IP
  * Open the tumbller-esp32s3 in a VSCode window with PlatformIO extension installed
  * On line 10 in the main.cpp file there is a macro called `// #define USE_SERIAL`. Comment/Uncomment to enable/disable serial output. For now enable it. Flash the ESP32-S3 with this firmware. 
  * On the serial terminal note down the IP of ESP32-S3
  * Now disable the serial output by commenting the macro `// #define USE_SERIAL` and flash the firmware again so that serial output is disabled
* Now create a file called .env in the `farcaster-frames-server` folder and paste this template content in that file

```
API_KEY=PAYBOT-KEY
DEBUG=False
ENVIRONMENT=development
CAMERA_URL_A=http://ESP32-S3-IP/getImage
CAMERA_URL_B=http://ESP32-S3-IP/getImage
BASE_URL=https://ngrok-ip.ngrok-free.app
TUMBLLER_URL_A=http://ESP-CAM-IP
TUMBLLER_URL_B=http://ESP-CAM-IP
MNEMONIC_ENV_VAR=FARCASTER-KEY
```
## Running the Mini App (aka frame server v2)

### Quickstart

A standard setup script can be used on Debian-based machines to get started. Please refer to it for other systems at this point, or to the detail below.

    bash scripts/setup.sh

Under the standard setup:

* `.env` needs to be edited to add DNS/FQDN name (e.g. tumbllers.yakrover.com) and options.
* Need to get the [Farcaster association file](https://farcaster.xyz/~/developers/mini-apps/manifest) to authorise this instance.
* `./bin/miniapp` starts the server to listen on all network interfaces. 
* [Optional] `./bin/fakerover` to start a fake (software) rover for demo and testing.
* [Optional] `./bin/dnsfrontend` to start the existing Ngrok DNS service to expose to the Internet.

### Detail

* Replace each of the variable with the correct value from `.env.template` and put them into a `.env` file.
* Get the association file from the Manifest tool: https://farcaster.xyz/~/developers/mini-apps/manifest
* Use Ngrok (or similar) to expose the server.
  * `ngrok http --url=<FQDN domain name in .env> 8080`
* Now setup the server code
  * `python -m venv venv`
  * `source venv/bin/activate`
  * `pip install ".[prod]"
* If you have no Tumbller around, you can use the fake rover
  * From the root of this repository (in a different shell): `PYTHONPATH=. python dev/fake_rover.py`
  * The fake rover by default listens on localhost at 5001, and can be modified in `.env`.
* TLS configuration is now comulsory for MiniApp servers. Here example with [Let's Encrypt](https://certbot.eff.org/instructions?ws=other&os=snap) on Ubuntu:
  * `sudo snap install --classic certbot`: Prepare for certificate confguration.
  * `sudo ln -s /snap/bin/certbot /usr/bin/certbot`: End of preps.
  * `sudo certbot certonly --standalone`: Configure certificate and renewal automation.
  * `sudo certbot renew --dry-run`: Check renewal works fine.
* Start the MiniApp server by going into the `farcaster-frames-server` folder
  * `cd farcaster-frames-server`
  * `sudo uvicorn main:app --host 0.0.0.0 --port 443 --reload --ssl-keyfile /etc/letsencrypt/live/yakrover.com/privkey.pem --ssl-certfile /etc/letsencrypt/live/yakrover.com/fullchain.pem`
  * Note Farcaster expects the server to listen on 443, not customizable anymore.
* Open a browser logged in with farcaster and then go to this url to test the frame
  * https://farcaster.xyz/~/developers/mini-apps/embed (you can also use the Manifest Tool; you can also try directly in a browser, but there will be no Farcaster integration there (e.g. payment will not work)).
  * Paste the Ngrok URL into it and play with the robot
