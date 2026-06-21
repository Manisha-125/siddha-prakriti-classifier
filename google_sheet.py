import os
import json
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SPREADSHEET_ID = "1rjBd4Ij7DJNhjgLZrk-O6ir9Wjj8MG7Txt7UmZZWW8k"
SHEET_NAME = "Admin_18_Feature_Audit"

FEATURES = [
    "Eyes", "Nose", "Tongue", "Ears", "Skin", "Nails",
    "Hair", "Facial Structure", "Lips", "Jawline", "Cheeks",
    "Neck", "Shoulders", "Chest", "Arms", "Hands", "Legs", "Feet"
]

def get_service():
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]

    if os.path.exists("credentials.json"):
        creds = Credentials.from_service_account_file(
            "credentials.json",
            scopes=scopes
        )
    elif os.environ.get("GOOGLE_CREDENTIALS_JSON"):
        service_account_info = json.loads(os.environ["GOOGLE_CREDENTIALS_JSON"])
        creds = Credentials.from_service_account_info(
            service_account_info,
            scopes=scopes
        )
    else:
        raise FileNotFoundError("credentials.json or GOOGLE_CREDENTIALS_JSON not found.")

    return build("sheets", "v4", credentials=creds)

def create_admin_sheet():
    service = get_service()

    spreadsheet = service.spreadsheets().get(
        spreadsheetId=SPREADSHEET_ID
    ).execute()

    existing_sheets = [
        sheet["properties"]["title"]
        for sheet in spreadsheet["sheets"]
    ]

    if SHEET_NAME not in existing_sheets:
        service.spreadsheets().batchUpdate(
            spreadsheetId=SPREADSHEET_ID,
            body={
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "title": SHEET_NAME
                            }
                        }
                    }
                ]
            }
        ).execute()

        print(f"Created new sheet: {SHEET_NAME}")
    else:
        print(f"Sheet already exists: {SHEET_NAME}")

    headers = [
        "Timestamp",
        "Name",
        "Age",
        "Gender",
        "Location",
        "Language",
        "Final_Prediction",
        "Vata_Prob",
        "Pitta_Prob",
        "Kapha_Prob"
    ]

    for feature in FEATURES:
        headers.extend([
            f"{feature}_Response",
            f"{feature}_Option",
            f"{feature}_Value",
            f"{feature}_Dosha",
            f"{feature}_Keywords",
            f"{feature}_Gemini_Reason",
            f"{feature}_Input_Mode"
        ])

    for i in range(1, 6):
        headers.extend([
            f"Top_LIME_Feature_{i}",
            f"Top_LIME_Weight_{i}",
            f"Top_LIME_Impact_{i}"
        ])

    headers.append("Clinical_Narrative")

    service.spreadsheets().values().clear(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A1:ZZ1"
    ).execute()

    service.spreadsheets().values().update(
        spreadsheetId=SPREADSHEET_ID,
        range=f"{SHEET_NAME}!A1",
        valueInputOption="USER_ENTERED",
        body={"values": [headers]}
    ).execute()

    # Freeze header row + basic formatting
    sheet_metadata = service.spreadsheets().get(
        spreadsheetId=SPREADSHEET_ID
    ).execute()

    sheet_id = None
    for sheet in sheet_metadata["sheets"]:
        if sheet["properties"]["title"] == SHEET_NAME:
            sheet_id = sheet["properties"]["sheetId"]
            break

    requests = [
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {
                        "frozenRowCount": 1
                    }
                },
                "fields": "gridProperties.frozenRowCount"
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1
                },
                "cell": {
                    "userEnteredFormat": {
                        "textFormat": {
                            "bold": True
                        },
                        "backgroundColor": {
                            "red": 0.85,
                            "green": 0.92,
                            "blue": 1.0
                        }
                    }
                },
                "fields": "userEnteredFormat(textFormat,backgroundColor)"
            }
        }
    ]

    service.spreadsheets().batchUpdate(
        spreadsheetId=SPREADSHEET_ID,
        body={"requests": requests}
    ).execute()

    print("Admin sheet header structure created successfully.")

if __name__ == "__main__":
    create_admin_sheet()