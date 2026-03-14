import os
import base64
import json
from typing import TypedDict, List, Dict
from io import BytesIO
import tempfile

import streamlit as st
from PIL import Image
import numpy as np
import cv2

from dotenv import load_dotenv
from segment_anything import sam_model_registry, SamPredictor

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


# ===================== ENV =====================
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path, override=True)


# ===================== CSS =====================
DARK_TEXT_CSS = """
<style>
textarea[disabled] {
    -webkit-text-fill-color: black !important;
    color: black !important;
    opacity: 1;
}
</style>
"""
st.markdown(DARK_TEXT_CSS, unsafe_allow_html=True)


# ===================== STATE =====================
class DetectedObject(TypedDict):
    label: str
    bbox: List[float]  # normalized


class FrameDetection(TypedDict):
    frame_number: int
    timestamp: float
    detections: List[DetectedObject]


class AgentState(TypedDict):
    video_path: str
    total_frames: int
    fps: float
    frame_detections: Dict[int, List[DetectedObject]]  # frame_number -> detections
    current_frame: int
    final_annotation: str
    object_relationships: str  # NEW: Store relationships between objects
    ontology: dict  # NEW: Store generated ontology
    human_feedback: str


# ===================== GEMINI =====================
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    api_key=os.getenv("GOOGLE_API_KEY")
)

GRAPH = None


# ===================== SAM =====================
@st.cache_resource
def load_sam():
    ckpt = os.path.join(os.path.dirname(__file__), "sam_vit_b_01ec64.pth")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to("cpu")
    return SamPredictor(sam)


sam_predictor = load_sam()


# ===================== VIDEO HELPERS =====================
def extract_key_frames(video_path: str, num_frames: int = 4):
    """
    Extract key frames from video at regular intervals
    Returns: list of (frame_number, frame_array, timestamp)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    key_frames = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = frame_idx / fps
            key_frames.append((frame_idx, frame_rgb, timestamp))
    
    cap.release()
    return key_frames, total_frames, fps


def get_frame_at_index(video_path: str, frame_idx: int):
    """Get a specific frame from video"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def save_video_with_annotations(video_path: str, frame_detections: Dict, output_path: str, selected_label: str = None):
    """
    Create a new video with annotations overlaid
    If selected_label is provided, only highlight that object
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # If this frame has detections, draw them
        if frame_idx in frame_detections:
            detections = frame_detections[frame_idx]
            for det in detections:
                if selected_label is None or det["label"] == selected_label:
                    x1, y1, x2, y2 = det["bbox"]
                    x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, det["label"], (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()


# ===================== JSON HELPER =====================
def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
    text = text.strip(" \n`")
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array found")
    return text[start:end + 1]


# ===================== HELPERS =====================
def encode_image(img: np.ndarray) -> str:
    """Encode numpy array (frame) to base64"""
    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def denormalize_box(box, w, h):
    x1, y1, x2, y2 = box
    return np.array([
        int(x1 * w),
        int(y1 * h),
        int(x2 * w),
        int(y2 * h),
    ])


def foreground_points_from_box(box):
    """Generate multiple foreground points inside the box"""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    points = [
        [cx, cy],                          # center
        [x1 + (x2 - x1) * 0.25, cy],       # left-middle
        [x1 + (x2 - x1) * 0.75, cy],       # right-middle
        [cx, y1 + (y2 - y1) * 0.25],       # top-middle
        [cx, y1 + (y2 - y1) * 0.75],       # bottom-middle
    ]

    return np.array(points, dtype=int)


def background_points_from_box(box, padding=10):
    """Generate background points just outside the box"""
    x1, y1, x2, y2 = box

    points = [
        [x1 - padding, y1 - padding],
        [x2 + padding, y1 - padding],
        [x1 - padding, y2 + padding],
        [x2 + padding, y2 + padding],
    ]

    return np.array(points, dtype=int)


def choose_best_mask(masks, box):
    """Select the most likely correct mask"""
    x1, y1, x2, y2 = box
    box_area = (x2 - x1) * (y2 - y1)

    best_mask = None
    best_score = float("inf")

    for mask in masks:
        area = mask.sum()
        score = area / box_area

        if score < best_score:
            best_score = score
            best_mask = mask

    return best_mask


# ===================== NEW: RELATIONSHIP DETECTION =====================
def analyze_object_relationships(frame: np.ndarray, detections: List[DetectedObject]) -> List[str]:
    """
    Analyze relationships between detected objects in a frame using Gemini
    Returns list of relationship descriptions
    """
    if not detections or len(detections) < 2:
        return []
    
    # Create a description of detected objects
    objects_list = [det["label"] for det in detections]
    objects_description = ", ".join(objects_list)
    
    instruction = (
        f"In this image, the following objects have been detected: {objects_description}\n\n"
        "Analyze the spatial and contextual relationships between these objects.\n"
        "Describe relationships like:\n"
        "- person wearing jacket\n"
        "- person holding phone\n"
        "- person sitting on chair\n"
        "- cup on table\n"
        "- dog next to person\n"
        "- person wearing hat\n"
        "- person folding blanket\n\n"
        "Return ONLY a JSON array of relationship strings. Each string should be a simple, clear relationship.\n"
        "Format: [\"person wearing jacket\", \"person wearing hat\", \"cup on table\"]\n"
        "If no clear relationships exist, return an empty array: []"
    )

    prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": instruction 
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(frame)}"
                }
            }
        ]
    )

    try:
        response = model.invoke([prompt])
        raw = response.content
        
        # Try to extract JSON array
        relationships = json.loads(extract_json(raw))
        return relationships if isinstance(relationships, list) else []
    except Exception as e:
        st.warning(f"Could not analyze relationships in this frame: {e}")
        return []


# ===================== NEW: KNOWLEDGE GRAPH VISUALIZATION =====================
def create_knowledge_graph_html(ontology: dict) -> str:
    """
    Create an interactive knowledge graph visualization using vis.js
    Returns HTML string with embedded visualization
    """
    if not ontology or not ontology.get("ontology", {}).get("classes"):
        return None
    
    classes = ontology["ontology"]["classes"]
    
    # Build nodes and edges
    nodes = []
    edges = []
    
    # Color mapping for categories
    category_colors = {
        "Human": "#FF6B6B",
        "Animal": "#4ECDC4",
        "Vehicle": "#45B7D1",
        "Furniture": "#96CEB4",
        "Clothing": "#FFEAA7",
        "SportsEquipment": "#DFE6E9",
        "ElectronicDevice": "#74B9FF",
        "Structure": "#A29BFE",
        "NaturalObject": "#55EFC4",
        "Tool": "#FDA7DF",
        "Food": "#FFD93D",
        "Other": "#B2BEC3"
    }
    
    for cls in classes:
        node_id = cls["name"]
        category = cls.get("category", "Other")
        color = category_colors.get(category, "#95A5A6")
        
        # Create node
        nodes.append({
            "id": node_id,
            "label": node_id,
            "title": f"Category: {category}",
            "color": color,
            "shape": "box",
            "font": {"size": 14, "color": "#2C3E50"}
        })
        
        # Create edges from relationships
        relationships = cls.get("relationships", [])
        for rel in relationships:
            rel_type = rel.get("type", "related")
            target = rel.get("target", "")
            
            if target:
                edges.append({
                    "from": node_id,
                    "to": target,
                    "label": rel_type,
                    "arrows": "to",
                    "font": {"size": 10, "align": "middle"},
                    "color": {"color": "#7F8C8D"}
                })
    
    # Convert to JSON
    nodes_json = json.dumps(nodes)
    edges_json = json.dumps(edges)
    
    # Create HTML with vis.js
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            html, body {{
                width: 100%;
                height: 100%;
                overflow: hidden;
            }}
            #mynetwork {{
                width: 100%;
                height: 700px;
                border: 1px solid #E0E0E0;
                background-color: #FAFAFA;
                border-radius: 8px;
            }}
            body {{
                font-family: Arial, sans-serif;
                padding: 10px;
            }}
            .legend {{
                margin-top: 15px;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .legend-item {{
                display: inline-block;
                margin-right: 20px;
                margin-bottom: 8px;
            }}
            .legend-color {{
                display: inline-block;
                width: 20px;
                height: 20px;
                margin-right: 5px;
                border-radius: 3px;
                vertical-align: middle;
            }}
            .info {{
                background: #E3F2FD;
                padding: 12px;
                border-radius: 6px;
                margin-bottom: 15px;
                border-left: 4px solid #2196F3;
            }}
        </style>
    </head>
    <body>
        <div class="info">
            <strong>Interactive Knowledge Graph</strong> - 
            Click and drag nodes to rearrange • 
            Scroll to zoom • 
            Click nodes to highlight connections
        </div>
        <div id="mynetwork"></div>
        <div class="legend">
            <strong>Legend:</strong><br><br>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #FF6B6B;"></span>
                <span>Human</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #FFEAA7;"></span>
                <span>Clothing</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #74B9FF;"></span>
                <span>Electronic</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #4ECDC4;"></span>
                <span>Animal</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #96CEB4;"></span>
                <span>Furniture</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #DFE6E9;"></span>
                <span>Sports</span>
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background-color: #B2BEC3;"></span>
                <span>Other</span>
            </div>
        </div>
        
        <script type="text/javascript">
            // Create nodes and edges
            var nodes = new vis.DataSet({nodes_json});
            var edges = new vis.DataSet({edges_json});
            
            // Create a network
            var container = document.getElementById('mynetwork');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            
            var options = {{
                nodes: {{
                    borderWidth: 2,
                    borderWidthSelected: 3,
                    shadow: true,
                    font: {{
                        size: 14,
                        color: '#2C3E50'
                    }}
                }},
                edges: {{
                    width: 2,
                    shadow: true,
                    smooth: {{
                        type: 'cubicBezier',
                        forceDirection: 'horizontal',
                        roundness: 0.4
                    }}
                }},
                physics: {{
                    enabled: true,
                    barnesHut: {{
                        gravitationalConstant: -2000,
                        centralGravity: 0.3,
                        springLength: 150,
                        springConstant: 0.04,
                        damping: 0.09,
                        avoidOverlap: 0.5
                    }},
                    stabilization: {{
                        iterations: 200
                    }}
                }},
                interaction: {{
                    hover: true,
                    tooltipDelay: 200,
                    navigationButtons: true,
                    keyboard: true
                }}
            }};
            
            var network = new vis.Network(container, data, options);
            
            // Highlight connected nodes on click
            network.on("click", function(params) {{
                if (params.nodes.length > 0) {{
                    var nodeId = params.nodes[0];
                    var connectedNodes = network.getConnectedNodes(nodeId);
                    var connectedEdges = network.getConnectedEdges(nodeId);
                    
                    // Highlight logic
                    var allNodes = nodes.get({{returnType: "Object"}});
                    var updateArray = [];
                    
                    for (var nodeKey in allNodes) {{
                        if (allNodes.hasOwnProperty(nodeKey)) {{
                            var node = allNodes[nodeKey];
                            if (connectedNodes.indexOf(node.id) !== -1 || node.id === nodeId) {{
                                node.opacity = 1;
                            }} else {{
                                node.opacity = 0.3;
                            }}
                            updateArray.push(node);
                        }}
                    }}
                    nodes.update(updateArray);
                }}
            }});
            
            // Reset on background click
            network.on("deselectNode", function(params) {{
                var allNodes = nodes.get({{returnType: "Object"}});
                var updateArray = [];
                
                for (var nodeKey in allNodes) {{
                    if (allNodes.hasOwnProperty(nodeKey)) {{
                        allNodes[nodeKey].opacity = 1;
                        updateArray.push(allNodes[nodeKey]);
                    }}
                }}
                nodes.update(updateArray);
            }});
        </script>
    </body>
    </html>
    """
    
    return html


# ===================== NEW: ONTOLOGY GENERATION =====================
def generate_ontology(all_labels: set, relationships_text: str) -> dict:
    """
    Generate a structured ontology from detected objects and relationships
    """
    if not all_labels:
        return {"ontology": {"classes": []}}
    
    object_list = ", ".join(sorted(all_labels))
    relationship_list = relationships_text if relationships_text else "No relationships detected"
    
    prompt_text = f"""You are a knowledge graph construction system.
Your task is to generate a structured ontology strictly based on the provided detected objects and relationships.

CRITICAL RULES:
1. DO NOT invent new objects.
2. DO NOT add entities that are not in the detected object list.
3. DO NOT modify object names.
4. DO NOT hallucinate relationships.
5. Use ONLY the provided objects and relationships.
6. If information is insufficient, leave fields empty.
7. Output ONLY valid JSON. No explanations. No markdown.

INPUT:
Detected Objects:
{object_list}

Detected Relationships:
{relationship_list}

REQUIRED OUTPUT FORMAT:
{{
  "ontology": {{
    "classes": [
      {{
        "name": "ObjectLabel",
        "category": "HighLevelCategory",
        "parent_class": "OptionalParentClassOrNull",
        "attributes": [],
        "relationships": [
          {{
            "type": "relationship_type",
            "target": "TargetObjectLabel"
          }}
        ]
      }}
    ]
  }}
}}

GUIDELINES:
1. "category" should be a general semantic category like:
   - Human
   - Animal
   - Vehicle
   - Furniture
   - Clothing
   - SportsEquipment
   - ElectronicDevice
   - Structure
   - NaturalObject
   - Tool
   - Food
   - Other

2. parent_class should only be used if logically valid.
   Example:
   - person → Human
   - jacket → Clothing
   - ball → SportsEquipment

3. relationships must be extracted from provided relationship strings.
   Example:
   - "person holding phone" → type: "holding", target: "phone"

4. If no relationships exist for an object, use:
   "relationships": []

5. If no attributes are explicitly known, use:
   "attributes": []

6. Keep naming consistent with detected labels.

Return strictly JSON only."""

    try:
        response = model.invoke([HumanMessage(content=prompt_text)])
        raw = response.content
        
        # Try to extract JSON
        raw = raw.strip()
        
        # Remove markdown code blocks if present
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]) if len(lines) > 2 else raw
        
        raw = raw.strip().strip("```").strip()
        
        # Find JSON object
        start_idx = raw.find("{")
        end_idx = raw.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            json_str = raw[start_idx:end_idx + 1]
            ontology = json.loads(json_str)
            return ontology
        else:
            st.warning("Could not extract valid JSON from ontology response")
            return {"ontology": {"classes": []}}
            
    except Exception as e:
        st.warning(f"Failed to generate ontology: {e}")
        return {"ontology": {"classes": []}}


# ===================== AGENT NODE =====================
def detect_objects_in_frame(frame: np.ndarray, feedback: str = "") -> List[DetectedObject]:
    """Detect objects in a single frame using Gemini"""
    
    instruction = (
        "Analyze this video frame and return ONLY a JSON array.\n"
        "Each item must contain:\n"
        "- label (e.g. dog, ball, person)\n"
        "- bbox [x1,y1,x2,y2] normalized between 0 and 1\n"
    )

    if feedback:
        instruction += (
            "\nIMPORTANT HUMAN FEEDBACK:\n"
            f"{feedback}\n"
            "Make sure the output follows this feedback."
        )

    prompt = HumanMessage(
        content=[
            {
                "type": "text",
                "text": instruction 
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encode_image(frame)}"
                }
            }
        ]
    )

    response = model.invoke([prompt])
    raw = response.content

    try:
        detections = json.loads(extract_json(raw))
        return detections
    except Exception as e:
        st.error(f"Failed to parse Gemini JSON: {e}")
        st.code(raw)
        return []


def detect_objects(state: AgentState) -> AgentState:
    """Detect objects across key frames of the video"""
    st.info("Agent: Detecting objects in video frames with Gemini...")
    
    video_path = state["video_path"]
    feedback = state.get("human_feedback", "").strip()
    
    # Extract key frames for analysis
    key_frames, total_frames, fps = extract_key_frames(video_path, num_frames=4)
    
    frame_detections = {}
    all_labels = set()
    all_relationships = set()  # NEW: Collect all unique relationships
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (frame_num, frame, timestamp) in enumerate(key_frames):
        status_text.text(f"Analyzing frame {idx + 1}/{len(key_frames)} (Frame #{frame_num}, Time: {timestamp:.2f}s)")
        
        # Step 1: Detect objects
        detections = detect_objects_in_frame(frame, feedback)
        frame_detections[frame_num] = detections
        
        for det in detections:
            all_labels.add(det["label"])
        
        # Step 2: Analyze relationships between detected objects
        status_text.text(f"Analyzing relationships in frame {idx + 1}/{len(key_frames)}...")
        relationships = analyze_object_relationships(frame, detections)
        
        for rel in relationships:
            all_relationships.add(rel)
        
        progress_bar.progress((idx + 1) / len(key_frames))
    
    status_text.text("✅ Detection complete!")
    
    # Format relationships for display
    relationships_text = "\n".join([f"• {rel}" for rel in sorted(all_relationships)]) if all_relationships else "No relationships detected"
    
    # Generate ontology from detected objects and relationships
    status_text.text("🔄 Generating ontology...")
    ontology = generate_ontology(all_labels, relationships_text)
    status_text.text("✅ Analysis complete with ontology!")
    
    return {
        "frame_detections": frame_detections,
        "total_frames": total_frames,
        "fps": fps,
        "current_frame": list(frame_detections.keys())[0] if frame_detections else 0,
        "final_annotation": ", ".join(sorted(all_labels)),
        "object_relationships": relationships_text,  # Add relationships
        "ontology": ontology  # NEW: Add ontology
    }


# ===================== GRAPH =====================
def get_graph():
    global GRAPH
    if GRAPH is None:
        g = StateGraph(AgentState)
        g.add_node("detect", detect_objects)
        g.set_entry_point("detect")
        g.add_edge("detect", END)
        GRAPH = g.compile()
    return GRAPH


def run_agent(state):
    st.session_state.graph_state = get_graph().invoke(state)


def reset_session():
    st.session_state.clear()
    st.rerun()


# ===================== SESSION INIT =====================
if "graph_state" not in st.session_state:
    st.session_state.graph_state = None
if "video_uploaded" not in st.session_state:
    st.session_state.video_uploaded = False
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "show_feedback_dialog" not in st.session_state:
    st.session_state.show_feedback_dialog = False


# ===================== UI =====================
st.set_page_config(page_title="Gemini + SAM Video Annotator", layout="wide")
st.title("🎬 LangGraph-style Video Annotation")

with st.sidebar:
    st.header("Controls")
    st.button("🔄 Reset", on_click=reset_session, type="primary")
    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file and uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.session_state.video_uploaded = True
        st.session_state.graph_state = None
        
        # Save video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            st.session_state.video_path = tmp_file.name


col_video, col_flow = st.columns([1, 1.5])

with col_video:
    st.subheader("🎥 Input Video")
    if st.session_state.video_path:
        st.video(st.session_state.video_path)
    else:
        st.info("Upload a video to begin")


with col_flow:
    state = st.session_state.graph_state

    # Phase 1: Start annotation
    if st.session_state.video_uploaded and state is None:
        if st.button("🚀 Start Annotation", type="primary"):
            run_agent({
                "video_path": st.session_state.video_path,
                "frame_detections": {},
                "total_frames": 0,
                "fps": 0.0,
                "current_frame": 0,
                "final_annotation": "",
                "object_relationships": "",  # Initialize relationships
                "ontology": {},  # NEW: Initialize ontology
                "human_feedback": ""
            })
            st.rerun()

    # Phase 2: Display results and allow frame navigation
    elif state and state.get("frame_detections"):
        frame_detections = state["frame_detections"]
        total_frames = state["total_frames"]
        fps = state["fps"]
        
        st.markdown("### 🏷 Detected Objects Across Video")
        
        # Collect all unique labels
        all_labels = set()
        for detections in frame_detections.values():
            for det in detections:
                all_labels.add(det["label"])
        
        if not all_labels:
            st.warning("No objects detected in the video.")
        else:
            labels = sorted(list(all_labels))
            selected = st.selectbox("Select Object to Highlight", labels)
            
            # Frame navigation
            st.markdown("---")
            st.markdown("### 🎞 Frame Navigation")
            
            frame_nums = sorted(list(frame_detections.keys()))
            
            # Find frames that contain the selected object
            frames_with_object = [
                fn for fn in frame_nums 
                if any(det["label"] == selected for det in frame_detections[fn])
            ]
            
            if frames_with_object:
                st.info(f"Object '{selected}' appears in {len(frames_with_object)} analyzed frames")
                
                # Frame selector - only show slider if there are frames to select
                if len(frames_with_object) == 1:
                    # If only one frame, don't use slider, just use that frame
                    selected_frame_idx = frames_with_object[0]
                    st.caption(f"Frame {selected_frame_idx} ({selected_frame_idx/fps:.2f}s)")
                else:
                    # Multiple frames - show slider
                    selected_frame_idx = st.select_slider(
                        "Select Frame",
                        options=frames_with_object,
                        format_func=lambda x: f"Frame {x} ({x/fps:.2f}s)"
                    )
                
                # Load and display the selected frame
                frame = get_frame_at_index(st.session_state.video_path, selected_frame_idx)
                
                if frame is not None:
                    # Get detections for this frame
                    frame_dets = frame_detections[selected_frame_idx]
                    
                    # Filter to only selected object
                    selected_det = next(
                        (d for d in frame_dets if d["label"] == selected), 
                        None
                    )
                    
                    if selected_det:
                        # Apply SAM segmentation
                        sam_predictor.set_image(frame)
                        
                        h, w = frame.shape[:2]
                        box = denormalize_box(selected_det["bbox"], w, h)
                        
                        fg_points = foreground_points_from_box(box)
                        bg_points = background_points_from_box(box)
                        
                        point_coords = np.vstack([fg_points, bg_points])
                        point_labels = np.array(
                            [1] * len(fg_points) + [0] * len(bg_points)
                        )
                        
                        masks, _, _ = sam_predictor.predict(
                            box=box,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True
                        )
                        
                        mask = choose_best_mask(masks, box)
                        
                        # Apply green overlay
                        annotated_frame = frame.copy()
                        annotated_frame[mask] = (
                            annotated_frame[mask] * 0.5 + np.array([0, 255, 0]) * 0.5
                        ).astype(np.uint8)
                        
                        st.image(
                            annotated_frame,
                            caption=f"Frame {selected_frame_idx}: {selected} highlighted",
                            use_column_width=True
                        )
                    else:
                        st.image(frame, caption=f"Frame {selected_frame_idx}", use_column_width=True)
            else:
                st.warning(f"Object '{selected}' not found in analyzed frames")
        
        # ===================== NEW: DISPLAY OBJECT RELATIONSHIPS =====================
        st.markdown("---")
        st.markdown("### 🔗 Object Relationships")
        
        if state.get("object_relationships"):
            st.text_area(
                "Detected Relationships Between Objects",
                state["object_relationships"],
                height=150,
                disabled=True,
                help="AI-detected spatial and contextual relationships between objects in the video"
            )
        else:
            st.info("No relationships detected between objects")
        
        # ===================== NEW: DISPLAY ONTOLOGY =====================
        st.markdown("---")
        st.markdown("### 🧠 Generated Ontology (Knowledge Graph)")
        
        if state.get("ontology") and state["ontology"].get("ontology", {}).get("classes"):
            # Create tabs for different views
            tab1, tab2 = st.tabs(["📊 Visual Graph", "📄 JSON Data"])
            
            with tab1:
                # Generate and display interactive knowledge graph
                st.markdown("#### Interactive Knowledge Graph Visualization")
                graph_html = create_knowledge_graph_html(state["ontology"])
                
                if graph_html:
                    # Display the interactive graph with increased height and width
                    st.components.v1.html(graph_html, height=850, scrolling=True)
                else:
                    st.warning("Could not generate graph visualization")
            
            with tab2:
                # Display as formatted JSON
                ontology_json = json.dumps(state["ontology"], indent=2)
                st.code(ontology_json, language="json")
                
                # Optional: Display summary
                num_classes = len(state["ontology"]["ontology"]["classes"])
                st.caption(f"Generated {num_classes} ontology classes from detected objects")
            
            # Add download button (visible in both tabs)
            st.download_button(
                label="📥 Download Ontology JSON",
                data=ontology_json,
                file_name="video_ontology.json",
                mime="application/json"
            )
        else:
            st.info("Ontology generation in progress or no objects detected")
        
        # ===================== HITL: Human Feedback (POPUP) =====================
        st.markdown("---")
        
        col_fb1, col_fb2 = st.columns(2)

        if col_fb1.button("🔁 Regenerate with Feedback"):
            st.session_state.show_feedback_dialog = True

        if col_fb2.button("✅ Accept Result"):
            st.success("Annotation accepted.")
            st.markdown("### ✅ Final Annotation")
            st.text_area(
                "Detected Objects",
                state["final_annotation"],
                height=120,
                disabled=True
            )

# ===================== FEEDBACK MODAL =====================
if st.session_state.get("show_feedback_dialog", False):

    @st.dialog("Provide Human Feedback")
    def feedback_dialog():
        feedback_text = st.text_area(
            "What should the agent improve?",
            placeholder="e.g. detect soccer players more accurately, include the ball",
            height=100
        )

        col_d1, col_d2 = st.columns(2)

        if col_d1.button("🚀 Submit & Regenerate", type="primary"):
            st.session_state.show_feedback_dialog = False

            run_agent({
                "video_path": st.session_state.video_path,
                "frame_detections": {},
                "total_frames": 0,
                "fps": 0.0,
                "current_frame": 0,
                "final_annotation": "",
                "object_relationships": "",  # Reset relationships
                "ontology": {},  # NEW: Reset ontology
                "human_feedback": feedback_text
            })

            st.rerun()

        if col_d2.button("❌ Cancel"):
            st.session_state.show_feedback_dialog = False
            st.rerun()

    feedback_dialog()