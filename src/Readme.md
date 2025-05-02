# Source Code Folder
To be structured as needed by project team.

Please document here

## Project Structure Overview

| **Subdirectory Name** | **Description** |
|-----------------------|-----------------|
| **Frontend** | This directory holds all the files for the frontend. The main file is located at: `src/HVACPredictor.jsx`. <br><br>To view the dashboard in action, run:<br><br>`npm run dev`<br><br>This will display a link (e.g., `http://localhost:5173/`) to the dashboard. However, the buttons will not function yet—this is because the backend needs to be running. |
| **Backend** | This directory contains the API file that connects the frontend to the backend.<br><br>Navigate to the file with:<br><br>`cd backend/api.py`<br><br>Run it using:<br><br>`python api.py`<br><br>Once the server is running, return to the frontend dashboard. From there, you can click **Train Model** and **Run Test** on any dataset of your choice.<br><br>![Dashboard](frontend/src/assets/dashboard.png) |

| | |
| | |
| | |
| | |
