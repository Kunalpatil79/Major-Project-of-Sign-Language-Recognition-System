import asyncio
import base64
import cv2
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Allow all origins for CORS (Cross-Origin Resource Sharing)
# This is important for browser-based clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origin
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Initialize with max_num_hands=1 for the demo
hands = mp_hands.Hands(
    max_num_hands=1,  # We'll focus on one hand for simplicity
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

def decode_image(base64_string):
    """Converts a base64 string (from the frontend) into a CV2 image."""
    try:
        img_bytes = base64.b64decode(base64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def process_hand_landmarks(image):
    """
    Processes an image to find hand landmarks using MediaPipe.
    Returns the processed image and landmark data.
    """
    # Flip the image horizontally for a later selfie-view display
    # and convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image_rgb)
    
    landmarks_data = None
    prediction = "None" # Default prediction

    # Draw the hand annotations on the image.
    if results.multi_hand_landmarks:
        # We process only the first hand found (due to max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]
            
        # Convert landmarks to a simple list of dicts for JSON serialization
        landmarks_data = []
        for landmark in hand_landmarks.landmark:
            # We must flip the x-coordinate back because the
            # frontend video is mirrored. 1.0 - landmark.x
            # This makes the landmarks align with the user's mirrored view.
            landmarks_data.append({"x": 1.0 - landmark.x, "y": landmark.y, "z": landmark.z})

        # --- Simple Finger Counting Logic ---
        # This is our "demo" model.
        # This part counts your fingers!
        prediction = count_fingers(hand_landmarks, results.multi_handedness[0])

    return image, landmarks_data, prediction

def count_fingers(hand_landmarks, handedness):
    """
    A simple rule-based finger counter.
    This is the "demo" model.
    """
    try:
        landmarks = hand_landmarks.landmark
        
        # Get hand label (Left or Right)
        hand_label = handedness.classification[0].label

        # Tip IDs
        tip_ids = [4, 8, 12, 16, 20]
        fingers = []

        # --- Thumb ---
        # Logic needs to account for hand orientation (Left vs Right)
        # We check if the thumb tip is further "out" than the joint below it.
        if hand_label == "Right": # Note: MediaPipe detects "Right" hand as user's actual right hand
            # For a right hand, "out" means a smaller x-coordinate
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)
        else: # Left hand
            # For a left hand, "out" means a larger x-coordinate
            if landmarks[tip_ids[0]].x > landmarks[tip_ids[0] - 1].x:
                fingers.append(1)
            else:
                fingers.append(0)

        # --- Other 4 Fingers ---
        # Check if the tip of the finger is above the joint below it (smaller y-coordinate)
        for id in range(1, 5): # For index, middle, ring, pinky
            if landmarks[tip_ids[id]].y < landmarks[tip_ids[id] - 2].y:
                fingers.append(1)
            else:
                fingers.append(0)
        
        total_fingers = sum(fingers)
        
        # --- Simple Sign Logic ---
        if total_fingers == 0:
            return "Fist"
        elif total_fingers == 1 and fingers[1] == 1: # Index finger only
            return "One"
        elif total_fingers == 2 and fingers[1] == 1 and fingers[2] == 1: # Index and Middle
            return "Two (Peace)"
        elif total_fingers == 3 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1: # Index, Middle, Ring
            return "Three"
        elif total_fingers == 4 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1: # All but thumb
            return "Four"
        elif total_fingers == 5:
            return "Five (Open Hand)"
        else:
            # Return the count if it's not a recognized simple sign
            return f"Fingers: {total_fingers}"

    except Exception as e:
        print(f"Error counting fingers: {e}")
        return "None"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """The main WebSocket endpoint for real-time recognition."""
    await websocket.accept()
    print("Client connected.")
    try:
        while True:
            # Receive base64 image data from the client
            data = await websocket.receive_text()
            
            # Decode the image
            image = decode_image(data)
            
            if image is None:
                continue

            # Process the image with MediaPipe
            processed_image, landmarks, prediction = process_hand_landmarks(image)
            
            # Send the prediction and landmarks back to the client
            response = {
                "prediction": prediction,
                "landmarks": landmarks # Sending landmarks so frontend can draw them
            }
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("Client disconnected.")
    except Exception as e:
        print(f"An error occurred: {e}")
        await websocket.close(code=1011) # 1011 = Internal Error


@app.get("/")
def read_root():
    return {"message": "Sign Language Recognition Backend is running. Connect via WebSocket to /ws"}

# To run this server:
# 1. Open your terminal
# 2. Navigate to the 'backend' folder
# 3. Run: uvicorn server:app --reload
if __name__ == "__main__":
    uvicorn.run("server:app", host="localhost", port=8000, reload=True)