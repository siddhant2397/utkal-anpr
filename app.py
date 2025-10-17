import streamlit as st
from mindee import ClientV2, InferenceParameters
from PIL import Image
import tempfile
import os
import json
import re
import pymongo
from datetime import datetime
import pytz
import pandas as pd

# MongoDB setup
api_key = st.secrets["MINDEE_API_KEY"]
mongo_uri = st.secrets["MONGODB_URI"]
client = pymongo.MongoClient(mongo_uri)
db = client["anpr_database"]
entry_collection = db["entry_vehicles"]
exit_collection = db["exit_vehicles"]
model_id = "6673789a-7589-4027-b9e2-6995470dbeab"

# Helper for inference
def run_inference(uploaded_file):
    file_ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
        tmp_file.write(uploaded_file.read())
        input_path = tmp_file.name
    try:
        mindee_client = ClientV2(api_key)
        params = InferenceParameters(model_id=model_id, rag=False)
        input_source = mindee_client.source_from_path(input_path)
        response = mindee_client.enqueue_and_get_inference(input_source, params)
        data = json.loads(response.raw_http) if isinstance(response.raw_http, str) else response.raw_http
        plate_val = (
            data.get("inference", {})
            .get("result", {})
            .get("fields", {})
            .get("license_plate_number", {})
            .get("value", None)
        )
        if plate_val:
            plate_val_uniform = re.sub(r'[^A-Za-z0-9]', '', plate_val).upper()
        else:
            plate_val_uniform = None
    finally:
        os.remove(input_path)
    return plate_val, plate_val_uniform

st.title("Utkal ANPR System")
tab_entry, tab_exit, tab_dashboard = st.tabs(["Entry", "Exit", "Dashboard"])

# Entry Tab
with tab_entry:
    file_in = st.file_uploader("Upload entry image (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"], key="entry_upload")
    if file_in is not None:
        st.info("Running plate recognition...")
        with st.spinner("Processing Entry Image..."):
            plate_val, plate_val_uniform = run_inference(file_in)
        if plate_val_uniform:
            ist = pytz.timezone('Asia/Kolkata')
            timestamp_ist = datetime.now(ist).isoformat()
            entry_count = entry_collection.count_documents({"plate_number": plate_val_uniform})
            exit_count = exit_collection.count_documents({"plate_number": plate_val_uniform})
            if entry_count > exit_count:
                st.error(f"❌ Cannot record entry: Vehicle {plate_val_uniform} has not exited yet!")
            else:
                entry_collection.insert_one({
                    "timestamp": timestamp_ist,
                    "plate_number": plate_val_uniform,
                    "original_plate": plate_val,
                })
            st.success(f"Entry Logged: {plate_val_uniform} at {timestamp_ist} IST")
        else:
            st.error("No license plate detected.")

# Exit Tab
with tab_exit:
    file_out = st.file_uploader("Upload exit image (jpg/png/pdf)", type=["jpg", "jpeg", "png", "pdf"], key="exit_upload")
    if file_out is not None:
        st.info("Running plate recognition...")
        with st.spinner("Processing Exit Image..."):
            plate_val, plate_val_uniform = run_inference(file_out)
        if plate_val_uniform:
            ist = pytz.timezone('Asia/Kolkata')
            timestamp_ist = datetime.now(ist).isoformat()
            # Check for earlier 'entry' record
            entry_exists = entry_collection.find_one({"plate_number": plate_val_uniform})
            authorized = bool(entry_exists)
            exit_collection.insert_one({
                "timestamp": timestamp_ist,
                "plate_number": plate_val_uniform,
                "original_plate": plate_val,
                "authorized": authorized
            })
            if authorized:
                st.success(f"✅ AUTHORIZED EXIT for {plate_val_uniform}")
            else:
                st.error(f"❌ UNAUTHORIZED EXIT for {plate_val_uniform}")
            st.info(f"Exit logged at {timestamp_ist} IST.")
        else:
            st.error("No license plate detected.")

# Dashboard Tab
with tab_dashboard:
    st.subheader("Entry/Exit Status Dashboard")
    # Get all unique plate numbers
    plates_in = set(doc["plate_number"] for doc in entry_collection.find())
    plates_out = {}
    for doc in exit_collection.find():
        plates_out[doc["plate_number"]] = {"timestamp": doc.get("timestamp", ""), "authorized": doc.get("authorized", False)}
    all_plates = plates_in.union(plates_out.keys())
    dashboard_data = []
    for plate in all_plates:
        entry_status = "Recorded" if plate in plates_in else "Not Recorded"
        exit_info = plates_out.get(plate, {})
        exit_status = ("Flagged" if plate in plates_out and not plates_out[plate].get("authorized", False)
                       else ("Exited" if plate in plates_out else "Not Exited"))

        authorized = exit_info.get("authorized", "")
        exit_time = exit_info.get("timestamp", "")
        dashboard_data.append({
            "Plate Number": plate,
            "Entry Status": entry_status,
            "Exit Status": exit_status,
            "Exit Time": exit_time,
            "Authorized Exit": authorized if exit_status == "Exited" else ""
        })
    if dashboard_data:
        dashboard_df = pd.DataFrame(dashboard_data)
        st.dataframe(dashboard_df)
    else:
        st.info("No vehicles recorded yet.")
