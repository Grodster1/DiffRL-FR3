import xml.etree.ElementTree as ET

def strip_finger_mimic(urdf_xml):
    """Strips <mimic> from urdf file which is unavailable in DART"""
    root = ET.fromstring(urdf_xml)
    for joint in root.findall("joint"):
        if joint.get("name") == "fr3_finger_joint2":
            for mimic in joint.findall("mimic"):
                joint.remove(mimic)
    return ET.tostring(root, encoding="unicode")