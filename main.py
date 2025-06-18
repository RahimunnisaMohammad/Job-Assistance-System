from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import process

app = Flask(__name__)
CORS(app)

# Load CSV data
try:
    jobs_df = pd.read_csv('C:/Users/rahim/OneDrive/Documents/miniproject (3)/miniproject/JobData.csv')
    courses_df = pd.read_csv('C:/Users/rahim/OneDrive/Documents/miniproject (3)/miniproject/coursera.csv')
    print("Datasets loaded successfully.")
except Exception as e:
    print(f"Error loading datasets: {e}")
    jobs_df, courses_df = None, None

# Check if datasets are loaded
if jobs_df is not None and courses_df is not None:
    # Preprocess job skills
    jobs_df['Skills'] = jobs_df['Skills'].fillna('').apply(lambda x: x.lower())
    
    # Initialize TF-IDF Vectorizer and KNN model
    vectorizer = TfidfVectorizer(stop_words='english')
    job_skills_tfidf = vectorizer.fit_transform(jobs_df['Skills'])
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(job_skills_tfidf)

    def fuzzy_skill_match(user_skills, job_skills):
        matched_skills = []
        for skill in job_skills:
            match = process.extractOne(skill, user_skills)
            if match[1] >= 80:  # Match score threshold
                matched_skills.append(skill)
        return matched_skills

    def get_job_recommendations(user_skills):
        user_skills_text = ' '.join(user_skills).lower()
        user_skills_tfidf = vectorizer.transform([user_skills_text])
        
        # Use KNN to find the nearest jobs based on user skills
        distances, indices = knn.kneighbors(user_skills_tfidf, n_neighbors=5)
        job_recommendations = []

        for idx in indices[0]:
            job = jobs_df.iloc[idx]
            job_skills = [skill.strip() for skill in job['Skills'].split(',')]
            matched_skills = fuzzy_skill_match(user_skills, job_skills)

            # Check if the user meets all required skills using fuzzy matching
            if len(matched_skills) >= len(job_skills):  # Match at least some skills
                job_recommendations.append({
                    'job_title': job['JobTitles'],
                    'job_link': job['Links'],
                    'job_skills': ', '.join(job_skills)
                })

        return job_recommendations

    def get_course_recommendations(missing_skills):
        course_recommendations = []

        for _, course in courses_df.iterrows():
            # Check if the course title has any missing skills
            course_title = course['course_title'].lower()
            if any(skill.lower() in course_title for skill in missing_skills):
                course_recommendations.append({
                    'course_title': course['course_title'],
                    'course_link': course['course_url']
                })

        return course_recommendations

    @app.route('/api/recommendations', methods=['POST'])
    def get_recommendations():
        data = request.json
        job_title = data.get('jobTitle', '').strip().lower()
        user_skills = [skill.strip().lower() for skill in data.get('skills', [])]

        print(f"Received request for job title: {job_title} with user skills: {user_skills}")

        # Find the exact matching job title
        exact_job_match = jobs_df[jobs_df['JobTitles'].str.lower() == job_title]

        if not exact_job_match.empty:
            required_skills = [skill.strip().lower() for skill in exact_job_match.iloc[0]['Skills'].split(',')]
            missing_skills = [skill for skill in required_skills if skill not in user_skills]

            print(f"Required skills for {job_title}: {required_skills}")
            print(f"Missing skills: {missing_skills}")

            if missing_skills:
                # Recommend courses for missing skills
                course_recommendations = get_course_recommendations(missing_skills)
                return jsonify({'courses': course_recommendations})
            else:
                # User has all required skills, recommend the job with the exact link
                job_recommendation = {
                    'job_title': exact_job_match.iloc[0]['JobTitles'],
                    'job_link': exact_job_match.iloc[0]['Links'],
                    'job_skills': ', '.join(required_skills)
                }
                return jsonify({'jobRecommendations': [job_recommendation]})

        else:
            # If no exact match, we can apply KNN to suggest jobs
            job_recommendations = get_job_recommendations(user_skills)
            if job_recommendations:
                return jsonify({'jobRecommendations': job_recommendations})
            else:
                return jsonify({'message': 'No suitable job recommendations found.'})

else:
    print("Error: Datasets were not loaded. Please check file paths.")

if __name__ == '__main__':
    app.run(debug=True)
