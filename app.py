from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, static_url_path='/static')

employee_path = 'data/employee_dataset2.csv'
employees_df = pd.read_csv(employee_path)

job_path = 'data/job_dataset.csv'
jobs_df = pd.read_csv(job_path)

employees_df["current_soft_skills"] = employees_df["current_soft_skills"].fillna("")  
employees_df["current_tech_stack"] = employees_df["current_tech_stack"].fillna("")
employees_df["soft_tech_combined"] = employees_df["current_soft_skills"] + " " + employees_df["current_tech_stack"]

jobs_df["SOFT_SKILLS_REQUIRED"] = jobs_df["SOFT_SKILLS_REQUIRED"].fillna("") 
jobs_df["TECHSTACK_REQUIRED"] = jobs_df["TECHSTACK_REQUIRED"].fillna("") 
jobs_df["soft_tech_combined"] = jobs_df["SOFT_SKILLS_REQUIRED"] + " " + jobs_df["TECHSTACK_REQUIRED"]

vectorizer = CountVectorizer()

employee_matrix = vectorizer.fit_transform(employees_df["soft_tech_combined"].astype(str))
job_matrix = vectorizer.transform(jobs_df["soft_tech_combined"].astype(str))

cosine_sim = cosine_similarity(employee_matrix, job_matrix)

def get_top_recommendations(emp_id, num_recommendations=3):
    if emp_id not in employees_df["emp_id"].values:
        print(f"Employee with emp_id {emp_id} not found.")
        return []

    emp_index = employees_df[employees_df["emp_id"] == emp_id].index[0]

    emp_scores = cosine_sim[emp_index]

    job_indices_scores = list(enumerate(emp_scores))

    job_indices_scores.sort(key=lambda x: x[1], reverse=True)

    current_job_index = employees_df[employees_df["emp_id"] == emp_id]["current_job"].index[0]
    job_indices_scores = [(index, score) for index, score in job_indices_scores if index != current_job_index]

    top_recommendations = []
    for index, score in job_indices_scores:
        job = jobs_df.iloc[index]

        current_salary_str = str(employees_df.iloc[emp_index]["salary"]).replace(",", "")
        job_salary_str = str(job["SALARY"]).replace(",", "")

        if current_salary_str and job_salary_str:
            current_salary = float(current_salary_str)
            job_salary = float(job_salary_str)

            salary_gap = abs(job_salary - current_salary)

            if salary_gap <= 0.1 * current_salary and job_salary != current_salary:
                recommended_soft_skills = set(map(str.strip, job['SOFT_SKILLS_REQUIRED'].split(',')))
                recommended_tech_stack = set(map(str.strip, job['TECHSTACK_REQUIRED'].split(',')))

                current_soft_skills = set(map(str.strip, employees_df.iloc[emp_index]['current_soft_skills'].split(',')))
                current_tech_stack = set(map(str.strip, employees_df.iloc[emp_index]['current_tech_stack'].split(',')))

                soft_skill_gap = recommended_soft_skills - current_soft_skills
                tech_stack_gap = recommended_tech_stack - current_tech_stack

                top_recommendations.append((index, {
                    'score': score,
                    'soft_skill_gap': soft_skill_gap,
                    'tech_stack_gap': tech_stack_gap
                }))

            if len(top_recommendations) == num_recommendations:
                break

    return top_recommendations

@app.route('/')
def index():
    background_image_url = app.static_url_path + '/demopic.jpeg'
    return render_template('index.html', background_image_url=background_image_url)

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    emp_id = request.form['emp_id']
    recommendations = get_top_recommendations(emp_id)
    
    employee_details = get_employee_details_by_id(emp_id)
    background_image_url1 = app.static_url_path + '/demopic.jpeg'

    return render_template('recommendations.html', emp_id=emp_id, employee_details=employee_details, recommendations=recommendations, jobs_df=jobs_df, background_image_url=background_image_url1)

def get_employee_details_by_id(emp_id):
    employee_details = employees_df[employees_df['emp_id'] == emp_id].to_dict(orient='records')

    if not employee_details:
        return None

    print(employee_details)

    return {
        'emp_id': employee_details[0]['emp_id'],
        'employee_name': employee_details[0]['employee_name'],
        'current_job': employee_details[0]['current_job'],
        'current_soft_skills': employee_details[0]['current_soft_skills'],
        'current_tech_stack': employee_details[0]['current_tech_stack'],
        'years_of_experience': employee_details[0]['years_of_experience'],
        'salary': employee_details[0]['salary']
    }

if __name__ == '__main__':
    app.run(debug=True)
