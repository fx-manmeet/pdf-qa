#

# Event Classification in Cricket with AI - Mini Project Report

# MINI PROJECT REPORT

# ON

# “Event Classification in Cricket with AI”

submitted by

- Dev Lad (21IT436)
- Shubham Prajapati (21IT445)
- Manmeet Patel (21IT456)

Faculty Guide: Prof. Priyank N. Bhojak

Information Technology Department

Birla Vishvakarma Mahavidyalaya Engineering College

(An Autonomous Institution)

AY: 2023-24, Semester II

WORK IS WORSHIP
---
#

# Event Classification in Cricket with AI

# Birla Vishvakarma Mahavidyalaya Engineering College

# Information Technology Department AY: 2023-24, Semester II

# CERTIFICATE

This is to certify that project entitled with "Event Classification in Cricket with AI" has been successfully carried out by Dev Lad (21IT436), Shubham Prajapati (21IT445), and Manmeet Patel (21IT456) for the subject of 3IT31-Mini Project under my guidance during the academic year 2023-24, Semester II. The Mini Project work carried out by the students of 6th semester is satisfactory.

Date:

Prof. Priyank N. Bhojak

Faculty Guide

IT Department, BVM

Dr. Nilesh Prajapati

Course Coordinator

IT Department, BVM

Dr. Keyur Bhrahmbhatt

Head of the Department

IT Department, BVM
---
#

# Event Classification in Cricket with AI - Acknowledgement

# ACKNOWLEDGEMENT

Special recognition to Prof. Priyank Bhojak, our faculty guide, whose steadfast leadership and continuous provision of invaluable feedback have been the cornerstone of our project's journey. His mentorship has not only enriched our academic experience but has also instilled a deeper understanding of our subject matter and a greater sense of confidence in our abilities.

Our sincere gratitude to Dr. Nilesh Parajapati, course coordinator, for choosing such an impressive project definition and excellent co-ordination throughout. Furthermore, we would like to express our gratitude to Keyur Bhrahmbhatt, HOD IT Dept., for his leadership and dedication in creating a positive academic environment.

Also grateful to any open source websites that provided us with cricket footage for our model's dataset, particularly YouTube. Also, ESPNCricInfo helped label the dataset. Furthermore, other research publications are referenced at the end of the study. Also, all the open-source libraries and frameworks are key cogs in the wheel.

Lastly, we thank all those who may have contributed to this project in ways that may have slipped our mind or gone unnoticed.

[Dev Lad, Shubham Prajapati, Manmeet Patel]
---
#

# Event Classification in Cricket with AI

# Table of Contents

# Chapter 1: Introduction

1.1 Aim of the Project - To classify events in cricket using AI

1.2 Project Scope - Scope of the project in terms of event classification

1.3 Project Objective - Objective of the project in achieving accurate event classification

1.4 Project Modules - Modules involved in the project for event classification

1.5 Project basic requirements - Basic requirements for the project implementation

# Chapter 2: Analysis, Design Methodology, and Implementation Strategy

2.1 Comparison of Existing Applications with Your Project - Comparison with existing solutions

2.2 Project Feasibility Study - Study on the feasibility of the project

2.3 Project Timeline Chart - Timeline for project development

2.4 Detailed Modules Description - Description of detailed project modules

2.5 Project SRS - Software Requirements Specification

- 2.5.1 Use Case Diagrams
- 2.5.2 Data Flow Diagrams
- 2.5.3 Entity Relationship Diagrams
- 2.5.4 Event Trace Diagram
- 2.5.5 State diagram
- 2.5.6 Class diagram

2.6 Database Design and Normalization - Design and normalization of the database
---
#

# Event Classification in Cricket with AI

# Chapter 3: Implementation and Testing

# 3.1 Software and Tools

Details about the software and tools used for the implementation.

# 3.2 User Interface and Snapshot

Information regarding the user interface design and snapshots of the application.

# 3.3 Testing Using Use Cases

Explanation of how testing was conducted using specific use cases.

# Chapter 4: Conclusion and Future Work

# 4.1 Conclusion

Summary of the conclusions drawn from the project.

# 4.2 Future Work

Discussion on potential future work and improvements for the project.

# References

List of references used in the project.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Abstract

This project showcases the creation of an application meant to improve viewer engagement and experience. In the realm of cricket broadcasting, the manual process of event classification and highlight generation presents limitations in speed, consistency, and personalization. To address this, we propose a novel approach utilizing deep learning for Ball by Ball Event Classification and Highlights Generation in cricket matches.

Our system aims to revolutionize the viewing experience by providing real-time event labeling, enabling dynamic highlight creation, and offering personalized content to users based on their preferences. Through a comprehensive system architecture, including modules for user registration, event classification, customized highlights generation, and prediction leader-board, our platform caters to the diverse needs of cricket enthusiasts.

By leveraging advanced technologies such as TensorFlow, PyTorch, and Keras for machine learning, coupled with React.js, MongoDB, and Express.js for front-end and back-end development, our system promises to deliver a seamless and engaging experience to users.

Through comparative analysis with traditional manual methods, our project highlights the advantages of automation, including enhanced precision, accelerated delivery, scalability, customization, economic efficiency, and uniform quality assurance.

Overall, our project represents a significant step towards modernizing cricket broadcasting, offering a transformative solution to enrich the viewing experience for cricket fans worldwide.
---
#

# Event Classification in Cricket with AI

# CHAPTER 1: Introduction

# 1.1 Aim of the Project

Cricket streaming rights sky-rockets every new season, but still there is very limited innovation here. The task of classifying and generating highlights is currently done manually, but could be automated to some extent to make the process more dynamic. With this project we plan to build a Ball by Ball Event Classification and Highlights generation from a cricket match using deep learning approach.

# 1.2 Project Scope

Our website caters to a diverse audience of cricket enthusiasts, including casual viewers seeking updated information and personalized highlights, sport analysts and commentators seeking deeper insights for informed commentary, and media and content creators, including social media influencers and journalists, looking for engaging content and discussion points to share with their followers.

# 1.3 Project Objective

We aim to label each moment of the match, making it easy to display events and revisit them later if missed in real-time. Additionally, we plan to optimize the system architecture to provide low-latency event classification. The highlights will be dynamically generated based on the identified key events, with consideration of the user's preferences.

# 1.4 Project Modules

# 1.4.1. User registration and Authentication:

This module is the entry point to a customized experience on our platform. Users can register and set up their profiles, which guarantees a safe login process. This module not only secures user information but also facilitates a personalized experience by enabling users to specify their preferences and keep track of their activity.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Event Classification and Display

At the heart of our platform, this module utilizes deep learning algorithms to classify live cricket events such as wickets, fours, sixes, and none. The classified events are displayed in real-time, offering users an insightful breakdown of the ongoing match. This feature enriches the viewing experience and helps catch up on important events.

# Customized Highlights Generation

Catering to diverse user preferences, this module allows users to generate personalized match highlights. Whether it's a compilation of every boundary in a match or a focus on key wickets, users can create their own highlight reels. This not only enhances engagement but also allows fans to relive their favorite moments from the match as per their interests.

# Prediction Leader-board

Engaging users in a fun and competitive way, the Prediction Leader-board module lets users predict outcomes of live match events. The predicted answers are compared with classified events by our algorithm. Correct predictions earn points, placing users on a leader-board where they compete with fellow cricket enthusiasts. This module adds an interactive layer to match viewing and creates opportunities for sponsors for their giveaways.

# Project Basic Requirements

# Hardware Requirements:

- A system with a minimum of 8GB RAM.
- A Multi-core processor, intel® i-7 4-core.
- Intel® UHD Graphics 620, for training classification algorithm video processing.

# Software Requirements:

- React.js framework for front-end development.
- MongoDB as a database management system.
- Express.js framework for back-end development.
- Git for version control.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Tech Stack Details:

- TensorFlow, PyTorch, and Keras Frameworks for building networks.
- NumPy, Pandas Data Processing Libraries for handling datasets.

This document contains all the tech stack details, approaches used, diagrams, and many more project-specific details. It contains many diagrams. Try to be precise while answering the questions.
---
#

# Event Classification in Cricket with AI

# CHAPTER 2: Analysis, Design Methodology, and Implementation Strategy

# 2.1 Comparison of Existing Applications with Your Project

# 2.1.1 From Subjectivity to Precision:

Traditional manual methods rely on human interpretation, which can introduce variability in event classification. Our automated system, on the other hand, uses deep learning to ensure objective and consistent accuracy, thereby enhancing the quality of event recognition.

# 2.1.2 Speed of Delivery:

The manual creation of highlights is a slow and labor-intensive process. Conversely, our automated approach promises near-instant highlight generation, transforming hours of footage into captivating moments in real-time.

# 2.1.3 Scaling New Heights:

Manual processing struggles to keep pace with multiple simultaneous matches, as it is limited by human capacity. Our system breaks these boundaries, effortlessly scaling to accommodate numerous games.

# 2.1.4 Customized Viewing Experience:

Whereas manual methods offer a one-size-fits-all approach, our system stands out by personalizing content. It uses viewer preference data to tailor highlights to individual tastes, thereby deepening viewer engagement.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# 2.1.5. Economic Efficiency:

The shift from manual labor to an automated system translates into significant cost savings. While the initial investment focuses on technology development, the long-term efficiency and reduced reliance on manual processes make it economically advantageous.

# 2.1.5. Uniform Quality Assurance:

Human-dependent processes inherently vary in consistency. Our automated system, immune to such fluctuations, promises a reliable and uniform quality of output, ensuring every highlight is of top-notch quality.

# 2.2 Project Feasibility Study

# 2.2.1. Technical Feasibility:

This involves the ability to implement and fine-tune deep learning models for accurate cricket event classification through research and development. Additionally, the handling of high-quality cricket match data for model training and validation needs to be evaluated. The availability of necessary hardware like GPUs for deep learning and software like programming languages and frameworks also needs to be considered.

# 2.2.2. Legal Feasibility:

This includes understanding and complying with legal restrictions related to broadcasting and replaying cricket match footage. Adhering to applicable data protection laws in handling user data, especially if personalization features are incorporated, is also necessary. Preparing for legal agreements necessary for content access and partnerships is also essential.

# 2.2.3. Market Feasibility:

Market demand analysis needs to be conducted to investigate the demand for AI-driven sports event classification and highlight generation among cricket audiences and media outlets. Identifying existing market solutions and positioning our project in terms of innovation and value addition is also important. Exploring collaboration possibilities with sports media entities.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Operational Feasibility:

This includes designing a user interface that is intuitive for both general users seeking match highlights and administrators overseeing system operations. It also involves ensuring that the system is reliable, especially during peak times like live matches, and establishing a robust maintenance plan. Planning for potential scaling needs to accommodate large user bases and high data volumes during major tournaments is also essential.

# Project Timeline Chart
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# 2.4 Detailed Modules Description

# 2.4.1. User registration and Authentication:

This module is the entry point to a customized experience on our platform.

# Functions:

- Users can register and set up their profiles, which guarantees a safe login process.
- This module not only secures user information but also facilitates a personalized experience by enabling users to specify their preferences and keep track of their activity.

# 2.4.2. Event classification and display:

At the heart of our platform, this module utilizes deep learning algorithms to classify live cricket events such as wickets, fours, sixes, and none.

# Functions:

- The classified events are displayed in real-time, offering users an insightful breakdown of the ongoing match.
- This feature enriches the viewing experience and helps users catch up on important events.

# 2.4.3. Customized highlights generation:

This module caters to diverse user preferences, allowing users to generate personalized match highlights.

# Functions:

- Users can create their own highlight reels, whether it's a compilation of every boundary in a match or a focus on key wickets.
- This enhances engagement and allows fans to relive their favorite moments from the match as per their interests.

# 2.4.4. Prediction Leader-board:

Engaging users in a fun and competitive way, the Prediction Leader-board module lets users predict outcomes of live match events.

# Functions:

- The predicted answers are compared with classified events by our algorithm. Correct predictions earn points, placing users on a leader-board where they compete with fellow cricket enthusiasts.
- This module adds an interactive layer to match viewing and creates opportunities for sponsors for their giveaways.
---
#

# Event Classification in Cricket with AI - Project SRS

# Project SRS - Use Case Diagrams

This diagram illustrates the various use cases in the Cricket Event Classification System:
- Login
- Authenticate
- Register
- Browse content
- Search content
- Stream content
- Predict outcome
- Generate highlights
- Customize highlights
- Save content
- DB Admin
---
#

# Event Classification in Cricket with AI

# Data Flow Diagram

# Level-0:

This level includes registration details, content order, cricket event classification system, verify details, content requirements, and database.

# Level-1:

This level involves registration or login credentials, authentication, verification feedback, match specifics for generation, customized highlights, match prediction, match details, prediction results, and databases.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Level-2:

Clippings Requirements:

- Parameters for Highlights 3.1
- Parameters for Highlights 3.2
- Parameters for Highlights 3.3

# User:

Verily choose custom highlights

Database match ordered clippings

Figure 2.5.2: DFD Level 2

# Entity Relationship Diagrams

Figure 2.5.3: E-R Diagram

Entities:

- User: uname, email, password
- Access: Kaich, Katch
- Players: Leaceroar, scores
- Match: contains tournament, venue
- Highlights: duration, implied_duration
- Inning: specifics
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# 2.5.4 Event Trace Diagram

Cricket event Classification System interacts with the Database through various steps as shown in the diagram:

- User requests registration
- Registration success
- User requests login
- Login success
- Send content order
- Process to create content
- Send content
- User logs out
- Confirm logout

Figure 2.5.4: Event Trace Diagram
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# State Diagram

This state diagram illustrates the flow of events in the cricket event classification system. It shows the different states and transitions involved in the process.

Key actions in the state diagram include:

- Clicking on "Sign In" and entering valid credentials to log in.
- Browsing matches and selecting a match.
- Clicking on "Generate Highlights" to view labeled clippings.
- Providing credentials to watch customized highlights.
- Predicting the next ball and streaming the match.
- Saving content for future reference.

Figure 2.5.5: State Diagram
---
#

# Event Classification in Cricket with AI

# Class Diagram

This class diagram represents the relationships between different entities in the system:
- User has attributes like name, email, password and can perform actions like register, login, logout.
- Leaderboard contains information about total players, current leader and methods to get top leaders and standings.
- Content has attributes like content ID, title, duration and methods like play, pause, stop.
- Saved entity deals with saving and managing content, with methods to save, remove saved content and manage storage.
- Match entity includes details about tournaments, matches, highlights and allows customization of match details and highlights.
---
#

# Event Classification in Cricket with AI

# Database Design and Normalization

# Table Name: Match

Description: to store match details

Primary Key: Match_id

Sr. No.
Name
Data Type
Constraint
Description

1
Match_id
INT
Primary Key
Unique match id

2
Match_tournament
VARCHAR(50)
Not Null
Store the name of tournament

3
Match_name
VARCHAR(100)
Not Null
Match’s unique name

4
Match_date
DATE
Not Null
Match’s date

5
Match_venue
VARCHAR(100)
Not Null
Venue of the match

# Table Name: Classified event

Description: To store information of classified event

Primary Key: Clip_id

Sr. No.
Name
Data Type
Constraint
Description

Match_id
INT
Foreign Key
Unique match id

Clip_id
INT
Primary Key
Unique id of every clip

Start_time
INT
-
Starting time of clip

End_time
INT
-
Ending time of clip

Event
VARCHAR
-
Name of classified event
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Table Name: User

Description: to store user credentials

Primary Key: Email_id

Sr. No.
Name
Data Type
Constraint
Description

1
User_name
VARCHAR(50)
Unique
Unique identifier for the user

2
Email_id
VARCHAR(50)
Primary Key
Email address of the user

3
Password
VARCHAR
Not Null
User’s Password

# Table Name: Leader Board

Description: to store information regarding leader board winnings

Foreign Key: Match_id

Sr. No.
Name
Data Type
Constraint
Description

1
Match_id
INT
Foreign Key
Unique match id

2
User_name
VARCHAR(100)
Not Null
Unique user identity

3
Points
INT
-
Points user scored
---
#

# Event Classification in Cricket with AI - Implementation and Testing

# CHAPTER 3: Implementation and Testing

# 3.1 Software and tools

# 3.1.1 Front-End Development

For the frontend development of the cricket event classification and highlights generation platform, React.js is utilized as the primary framework due to its component-based architecture and efficient rendering. Alongside React.js, HTML, CSS, and JavaScript (ES6+) were employed for structuring web pages, styling the user interface, and adding interactivity. Additionally, Redux is used for state management, and Bootstrap streamlined development with pre-designed components and responsive layouts.

# 3.1.2 Back-End Development

Express.js served as the framework for building RESTful APIs and handling server-side logic, while MongoDB is used as the database management system for storing and retrieving data related to user profiles, match events, and preferences. Node.js provided the runtime environment for server-side JavaScript execution, and Git for version control. FFMpeg was employed for merging video files for highlights generation. For the requirement of continuously sending data to client from server for Server-Sent Events(SSE).

HTML:
---
#

# Event Classification in Cricket with AI

# User Interface and Snapshot

# Login

The login page features essential fields for users to input their email address and password, along with a prominent "Submit" button. Once users enter their credentials and click "Submit," the data is sent to the Mongoose Compass, where it is securely stored in the database. This process ensures that user login information is safely managed and accessible for authentication purposes, contributing to a seamless and secure user experience on the platform.

Figure 3.2.1: Login System
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

3.2.2 Sign-up Page

The signup page provides users with fields to input their email address, password, and username, essential for creating a new account. Upon completion, users are redirected to the home page, where they can begin exploring the platform's features and content. This streamlined process ensures a smooth transition for new users, allowing them immediate access to the platform's offerings after registration.

Epic Cricket Content: Unleash the Excitement!

Explore the thrill of cricket matches right on our website! Dive into live updates, match highlights, expert analyses, and exclusive interviews with your favorite cricket stars. Whether it's the latest scores, captivating moments, or in-depth insights into the cricketing world - from exhilarating boundary shots to biting finishes, immerse yourself in the excitement of every match, right from the comfort of your screen.
---
#

# Event Classification in Cricket with AI

# Home-page

Step into the world of cricket with our website's inviting homepage, where a sleek navigation bar awaits to guide you seamlessly through our platform's offerings. Scroll down to discover captivating visuals and a concise description of our features. At the bottom, our footer houses all contact details, ensuring easy communication. Welcome to an immersive cricket experience designed just for you.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Event Classification and Display

The "Matches" section serves as a hub for users to access ongoing matches, displaying a list of current matches available for viewing. Upon selection of a specific match, users are redirected to the live match page where event classification is predicted in real-time.

Alongside the match classification, a side box displays the Prediction Leaderboard, engaging users in interactive match predictions. This dynamic feature enriches the viewing experience by providing live match updates and fostering user participation through predictions, enhancing overall engagement with the platform.

# Classification Model

The classification model employed in the platform analyzes cricket videos, categorizing each ball delivery into four distinct classes: "No Run," "Boundary," "Wicket," and "Run." This model operates by leveraging deep learning algorithms to identify and classify the outcomes of each ball delivery in real-time. By accurately distinguishing between different event types, including.

# Diagram: Event Classification and Display
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

Boundaries, wickets, and runs, the model enhances the platform's ability to provide users with comprehensive and informative match highlights, thereby enriching their viewing experience.

# Customize Highlights Generation

The Customized Highlights Generation feature empowers users to curate personalized match highlights tailored to their preferences. Users input their desired criteria, such as specific events like boundaries or wickets, and the platform generates customized highlight videos accordingly.

The screenshot showcases the user interface where users can select their preferences, with the adjacent section displaying the dynamically generated customized highlight videos in real-time. This interactive functionality enhances user engagement by allowing them to relive their favorite moments from the match according to their interests, thereby fostering a deeper connection with the game.

Figure 3.2.5: Customized Highlights
---
#

# Event Classification in Cricket with AI

# Leader Board

|Rank|Name|Email|Score|
|---|---|---|---|
|1|YOU|shubham123@gmail.com| |
|2|dev|devplad@gmail.com| |
|3|123rty|qogmail.com| |
|4|new7|new7@gmail.com| |
|5|new9|@gmail.com| |
|6|dnruv|dhruve@gmail.com| |
|7|daksha|Daskha@gmail.com| |
|8|qwe|f@gmail.com| |
|9|Iana|rodhes@gmail.com| |
|10|weld|e@gmail.com| |

To foster user participation, encourage community interaction, and create opportunities for sponsors to engage with the audience through giveaways and promotions, this feature is included. Here users can take a quiz on the website while streaming match, revisiting events or in between generating personalized highlights.

The leaderboard displays the usernames of participants along with their current point totals, ranking them in descending order based on their scores. Users can track their progress and compare their standings with other participants, fostering a sense of competition and camaraderie among users. The leaderboard is updated dynamically using technologies such as Server-Sent Events (SSE) or WebSocket connections.
---
#

# Event Classification in Cricket with AI

# Testing Using Use Cases

# User Authentication:

This test case focuses on the user authentication process, which includes sign-up and sign-in functionalities. The figure likely illustrates the flow of steps or the user interface elements involved in the authentication process, such as entering credentials, verifying information, and granting access to the application.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# Clipping and Classification

This test case covers the backend process of the application, specifically the clipping and classification of data or content. The figure might depict the workflow or the architecture of the backend system, including components responsible for segmenting or extracting relevant clips from a larger dataset and classifying or categorizing those clips based on specific criteria or models.

Figure 3.3.2: Clipping and Classification
---
#

# Event Classification in Cricket with AI

# Merging clips (For customized highlights)

This test case addresses the generation of customized highlights by merging individual clips. The figure is likely illustrating the process or the algorithm involved in combining multiple clips, potentially based on user preferences or predefined rules, to create a curated or personalized highlight reel or summary.

# Figure 3.3.3: Merging clips

OUTLINE

Merged Video Path: /Users/MaxNewton/Desktop/new2/backend/outputFolder/merged_1715687325064.mp4

TIMELINE

89 main" @ 0 4 3 Ln 347, Col Spaces: UTF-8 CRLF JavaScript Go Live
---
#

# Event Classification in Cricket with AI - Conclusion and Future Work

# CHAPTER 4: Conclusion and Future Work

# 4.1 Conclusion

Our application opened up new perspectives on cricket streaming. It also revealed AI's ability to transform many aspects of today's world for the betterment of humanity. Despite the challenges faced during this project, we are confident that with additional work, we can provide a more solid and production-ready solution. Our passion for this subject has driven us to explore and achieve our goals.

# 4.2 Future Work

Due to the complexity of this project, there are limitations in key parameters such as accuracy and latency. In the future, we aim to overcome these limitations by implementing the following:

- Expand the dataset by outsourcing to generate a larger and more diverse dataset to improve accuracy in event classification.
- Implement optimizations in the system architecture and data processing pipeline to minimize latency in event classification and highlight generation.
- Optimize algorithms for faster inference, utilize hardware acceleration like GPUs or specialized inference chips, and optimize database queries and network communication for real-time performance.
- Work towards productionizing the solution once it meets the required parameters, either independently or under a partnership.

These future works aim to enhance the accuracy and efficiency of the event classification system in cricket streaming.
---
#

# Event Classification in Cricket with AI

# Event Classification in Cricket with AI

# References

# Websites:

- Keras
- Numpy
- Cv2
- FFmpeg
- Chokidar
- Range
- MoviePy
- Shutil

# Research papers:

- Xingjian Shi, Zhourong Chen, Hao Wang, Dit-Yan Yeung, Wai-Kin Wong, Wang-chun Woo. Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting.
- Kalpit Dixit, Anusha Balakrishnan. Deep Learning using CNNs for Ball-by-Ball Outcome Classification in Sports.
- Kumail Abbas, Muhammad Saeed. Deep-Learning-Based Computer Vision Approach For The Segmentation Of Ball Deliveries And Tracking In Cricket.
