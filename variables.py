path="data/"
passage_file_name=path + "passage.pdf"
table_file_name=path + "tablee.pdf"
markdown_file_name=path + "markdownformat.md"#without table
table_markdown_file_name=path + "table_markdown.md"

old_eval_set = [
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

eval_set = [
    {
        "question": "Which society focuses on peer counseling and stress management?",
        "ground_truth": "Institute Wellness Group (IWG) focuses on peer counseling, stress management, and mental well-being."
    },
    {
        "question": "What does the society 'Sarth' offer?",
        "ground_truth": "Sarth provides a helpline and support system for students facing emotional distress or emergencies."
    },
    {
        "question": "What is the mission of Encore?",
        "ground_truth": "Encore promotes English language dramatics and represents IIT Kharagpur in drama events."
    },
    {
        "question": "What distinguishes WTMS from other music societies?",
        "ground_truth": "WTMS specializes in Western musical styles and performs in genres like rock, jazz, and blues."
    },
    {
        "question": "What is the focus area of the culinary society?",
        "ground_truth": "The Culinary Club promotes culinary skills through workshops, competitions, and community cooking."
    },
    {
        "question": "What is The Scholars’ Avenue known for?",
        "ground_truth": "An independent student newspaper that values editorial freedom and publishes campus news and insights."
    },
    {
        "question": "What kind of content does Awaaz publish?",
        "ground_truth": "Awaaz publishes Hindi-language news, campus updates, and student articles."
    },
    {
        "question": "What is InternPedia?",
        "ground_truth": "A centralized database in TSA with internship experiences and guidance for students."
    },
    {
        "question": "What is RoboSoccer?",
        "ground_truth": "A robotic competition where autonomous robots play soccer; KRSSG participates in such events."
    },
    {
        "question": "Which society works on swarm robotics?",
        "ground_truth": "The Swarm Project team under TRS focuses on developing swarm robotics."
    },
    {
        "question": "What is the function of the Quant Club?",
        "ground_truth": "The Quant Club promotes quantitative finance, data analytics, and algorithmic trading skills."
    },
    {
        "question": "What is the primary activity of Finterest?",
        "ground_truth": "Finterest focuses on financial markets, investment strategies, and economics-related discussions."
    },
    {
        "question": "What does 180 Degrees Consulting at IITKGP do?",
        "ground_truth": "It provides pro bono consulting to nonprofits and promotes social impact through strategy projects."
    },
    {
        "question": "What does iGEM IIT Kharagpur focus on?",
        "ground_truth": "iGEM IIT Kharagpur focuses on synthetic biology and represents the institute in iGEM competitions."
    },
    {
        "question": "What is the objective of the Science Education Group?",
        "ground_truth": "To foster interest in science among students and assist in competitive exam preparation."
    },
    {
        "question": "Which group supports UPSC aspirants?",
        "ground_truth": "The Public Policy and Governance Society supports UPSC aspirants with mentorship and resources."
    },
    {
        "question": "What activities does Technology Adventure Society (TAdS) organize?",
        "ground_truth": "TAdS organizes trekking, cycling, rock climbing, and other outdoor adventure activities."
    },
    {
        "question": "What is the purpose of the Technology Environment Society (TES)?",
        "ground_truth": "TES promotes sustainability, environmental awareness, and conducts eco-campaigns on campus."
    },
    {
        "question": "What does Ambar represent at IIT Kharagpur?",
        "ground_truth": "Ambar advocates for gender and sexual diversity and supports LGBTQIA+ students."
    },
    {
        "question": "What does the CRY chapter at IITKGP do?",
        "ground_truth": "The CRY chapter works for children's rights and organizes social campaigns and education drives."
    },
    {
        "question": "What ideology does the Ambedkar Periyar Study Circle (APSC) promote?",
        "ground_truth": "APSC promotes equality, social justice, and the ideologies of Ambedkar and Periyar."
    },
    {
        "question": "What is Think India's mission?",
        "ground_truth": "Think India promotes innovation, social awareness, and leadership among students."
    },
    {
        "question": "What does CodeStash aim to achieve?",
        "ground_truth": "CodeStash helps students improve competitive programming skills through practice and mentorship."
    },
    {
        "question": "What is the mission of KOSS?",
        "ground_truth": "The Kharagpur Open Source Society (KOSS) promotes open-source development and contributions."
    },
    {
        "question": "What does the CodeClub do?",
        "ground_truth": "CodeClub fosters coding interest through contests, projects, and peer learning sessions."
    },
    {
        "question": "What does the Chess Club do?",
        "ground_truth": "The Chess Club organizes tournaments, training sessions, and promotes chess culture on campus."
    },
    {
        "question": "What is the mission of the Standup and Improv Society?",
        "ground_truth": "To promote comedy, stand-up, and improvisational performance among students."
    },
    {
        "question": "What kind of content does the Rap Society produce?",
        "ground_truth": "The Rap Society creates original hip-hop content and promotes lyrical storytelling."
    },
    {
        "question": "What does the TEDx society do?",
        "ground_truth": "Organizes TEDx events to showcase innovative ideas and foster intellectual dialogue."
    },
    {
        "question": "What kind of students does NESF cater to?",
        "ground_truth": "NESF supports students from the North East, promoting cultural integration and awareness."
    }
]