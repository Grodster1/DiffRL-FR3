import xml.etree.ElementTree as ET

def strip_finger_mimic(urdf_xml):
    """franka_hand.xacro hardkoduje <mimic> na fr3_finger_joint2 (brak parametru
    do wyłączenia) — ros2_control odmawia command_interface na mimic joint.
    Opcja A: jawne sterowanie oboma palcami, więc usuwamy <mimic> po stronie
    wygenerowanego URDF, zamiast patchować franka_description."""
    root = ET.fromstring(urdf_xml)
    for joint in root.findall("joint"):
        if joint.get("name") == "fr3_finger_joint2":
            for mimic in joint.findall("mimic"):
                joint.remove(mimic)
    return ET.tostring(root, encoding="unicode")