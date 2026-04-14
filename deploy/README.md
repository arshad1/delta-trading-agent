# Ubuntu Production Deployment

This folder contains the configuration necessary to run the Trading Agent in the background on an Ubuntu server.

## 1. Setup the project
Make sure the project is cloned to your server (e.g., `/opt/hyperliquid-trading-agent`), and the virtual environment is set up:
```bash
# Clone repo
git clone <repo-url> /opt/hyperliquid-trading-agent
cd /opt/hyperliquid-trading-agent

# Set up python venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Setup your MySQL Database
Ensure that you have MySQL installed, created the `trading_agent` database, and modified your `.env` file with the correct credentials.

## 3. Install the systemd Service
Copy the `trading-agent.service` file to the systemd directory. 
*Note: Make sure to edit the `trading-agent.service` file first if your repository is not located at `/opt/hyperliquid-trading-agent` or if your username is not `ubuntu`.*

```bash
# Copy the service file
sudo cp deploy/trading-agent.service /etc/systemd/system/

# Reload the systemd daemon so it recognizes the newly added file
sudo systemctl daemon-reload

# Enable the service so it starts automatically on server boot
sudo systemctl enable trading-agent

# Start the service right now
sudo systemctl start trading-agent
```

## 4. Monitoring the application
Once the service is started, it runs in the background. You can check its status and view the logs at any time.

**Check the status:**
```bash
sudo systemctl status trading-agent
```

**View live logs (Tail):**
```bash
sudo journalctl -u trading-agent -f
```

## Stopping / Restarting
If you push new code or change the `gunicorn.conf.py`, you can easily restart the service:
```bash
sudo systemctl restart trading-agent
```
To stop the service entirely:
```bash
sudo systemctl stop trading-agent
```
