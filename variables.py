path="ilovepdf_split-range/"
passage_file_name=path + "passage.pdf"
table_file_name=path + "tablee.pdf"
markdown_file_name=path + "markdownformat.md"#without table
table_markdown_file_name=path + "table_markdown.md"

eval_set = [
    {
        "question": "What is the primary role of student societies at IIT Kharagpur?",
        "ground_truth": "Student societies enrich the campus experience by offering platforms for exploring interests, cultivating skills, and fostering leadership and social responsibility."
    },
    {
        "question": "What is the motto of the Technology Students' Gymkhana (TSG)?",
        "ground_truth": "Yogah Karmasu Kausalam"
    },
    {
        "question": "Which body acts as the hub for extracurricular activities at IIT Kharagpur?",
        "ground_truth": "Technology Students' Gymkhana (TSG)"
    },
    {
        "question": "Why do many societies at IIT Kharagpur use the prefix 'Technology' in their names?",
        "ground_truth": "It reflects a cultural tradition and institutional identity that fosters unity and shared pride among students."
    },
    {
        "question": "What is the mission of the Student Welfare Group (SWG)?",
        "ground_truth": "To help students develop their skills and personality, and ensure smooth functioning of their college life."
    },
    {
        "question": "Which society at IIT Kharagpur promotes gender and sexual diversity?",
        "ground_truth": "Ambar"
    },
    {
        "question": "What does the Eastern Technology Music Society (ETMS) focus on?",
        "ground_truth": "Performing popular Hindi numbers and promoting lesser-known bands through campus events."
    },
    {
        "question": "Which society represents IIT Kharagpur in Robosoccer competitions like RoboCup?",
        "ground_truth": "Kharagpur Robosoccer Students' Group (KRSSG)"
    },
    {
        "question": "What is the Scholars' Avenue known for?",
        "ground_truth": "It is an independent student-run newspaper that prioritizes editorial freedom over financial support."
    },
    {
        "question": "What does the Technology Adventure Society (TAdS) offer to students?",
        "ground_truth": "Expeditions and outdoor activities like trekking, rock climbing, and cycling."
    }
]