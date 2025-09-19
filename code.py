import json
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple
from datetime import datetime
import re
from collections import defaultdict
import sqlite3
import os

@dataclass
class Candidate:
    """Data model for internship candidates"""
    id: int
    name: str
    skills: List[str]
    qualifications: str
    location: str
    sector: str
    category: str  # General, OBC, SC, ST
    district: str
    past_participation: bool
    cgpa: float
    experience: str
    email: str = ""
    phone: str = ""
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Internship:
    """Data model for internship opportunities"""
    id: int
    company: str
    title: str
    required_skills: List[str]
    location: str
    sector: str
    capacity: int
    preferred_qualification: str
    stipend: int
    duration: str
    type: str  # Full-time, Part-time
    description: str = ""
    
    def to_dict(self):
        return asdict(self)

@dataclass
class Match:
    """Data model for candidate-internship matches"""
    candidate_id: int
    internship_id: int
    score: float
    factors: Dict[str, float]
    status: str
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

class AIMatchingEngine:
    """Core AI matching algorithm for internship placements"""
    
    def __init__(self):
        self.weights = {
            'skills': 0.40,
            'location': 0.20,
            'sector': 0.15,
            'qualification': 0.10,
            'diversity': 0.10,
            'academic': 0.05
        }
        
        # Priority districts for affirmative action
        self.aspirational_districts = [
            'Rural Karnataka', 'Sabarkantha', 'Adilabad', 'Dantewada', 
            'Naxalbari', 'Kalahandi', 'Nuapada', 'Koraput'
        ]
    
    def calculate_skills_match(self, candidate_skills: List[str], required_skills: List[str]) -> float:
        """Calculate skills matching score using fuzzy matching"""
        if not candidate_skills or not required_skills:
            return 0.0
        
        matches = 0
        total_required = len(required_skills)
        
        for req_skill in required_skills:
            for cand_skill in candidate_skills:
                # Exact match
                if req_skill.lower() == cand_skill.lower():
                    matches += 1
                    break
                # Partial match (contains)
                elif (req_skill.lower() in cand_skill.lower() or 
                      cand_skill.lower() in req_skill.lower()):
                    matches += 0.7
                    break
                # Similar skills mapping
                elif self._are_similar_skills(req_skill, cand_skill):
                    matches += 0.5
                    break
        
        return min(matches / total_required, 1.0)
    
    def _are_similar_skills(self, skill1: str, skill2: str) -> bool:
        """Check if two skills are similar using domain knowledge"""
        skill_groups = {
            'programming': ['python', 'java', 'javascript', 'c++', 'coding', 'programming'],
            'data': ['data analysis', 'analytics', 'sql', 'database', 'data science'],
            'design': ['ui/ux', 'figma', 'photoshop', 'design', 'prototyping'],
            'marketing': ['marketing', 'social media', 'content writing', 'seo'],
            'finance': ['finance', 'accounting', 'excel', 'financial modeling']
        }
        
        skill1_lower = skill1.lower()
        skill2_lower = skill2.lower()
        
        for group in skill_groups.values():
            if skill1_lower in group and skill2_lower in group:
                return True
        return False
    
    def calculate_location_score(self, candidate_location: str, internship_location: str) -> float:
        """Calculate location preference score"""
        if candidate_location.lower() == internship_location.lower():
            return 1.0
        
        # Same state bonus (simplified)
        state_mapping = {
            'delhi': 'delhi', 'mumbai': 'maharashtra', 'bangalore': 'karnataka',
            'hyderabad': 'telangana', 'ahmedabad': 'gujarat', 'pune': 'maharashtra',
            'kolkata': 'west bengal', 'chennai': 'tamil nadu'
        }
        
        cand_state = state_mapping.get(candidate_location.lower())
        intern_state = state_mapping.get(internship_location.lower())
        
        if cand_state and intern_state and cand_state == intern_state:
            return 0.7
        
        return 0.3  # Different state penalty
    
    def calculate_diversity_score(self, candidate: Candidate) -> float:
        """Calculate affirmative action/diversity score"""
        score = 0.0
        
        # Category-based scoring
        if candidate.category in ['SC', 'ST']:
            score += 0.5
        elif candidate.category == 'OBC':
            score += 0.3
        
        # District-based scoring
        if any(district.lower() in candidate.district.lower() 
               for district in self.aspirational_districts):
            score += 0.3
        
        if 'rural' in candidate.district.lower():
            score += 0.2
        
        # Past participation bonus
        if not candidate.past_participation:
            score += 0.2
        
        return min(score, 1.0)
    
    def calculate_match_score(self, candidate: Candidate, internship: Internship) -> Tuple[float, Dict[str, float]]:
        """Calculate overall match score between candidate and internship"""
        factors = {}
        
        # Skills matching
        skills_score = self.calculate_skills_match(candidate.skills, internship.required_skills)
        factors['skills'] = skills_score * 100
        
        # Location preference
        location_score = self.calculate_location_score(candidate.location, internship.location)
        factors['location'] = location_score * 100
        
        # Sector alignment
        sector_score = 1.0 if candidate.sector.lower() == internship.sector.lower() else 0.3
        factors['sector'] = sector_score * 100
        
        # Qualification relevance
        qual_words = internship.preferred_qualification.lower().split()
        cand_qual = candidate.qualifications.lower()
        qual_score = 1.0 if any(word in cand_qual for word in qual_words) else 0.5
        factors['qualification'] = qual_score * 100
        
        # Diversity score
        diversity_score = self.calculate_diversity_score(candidate)
        factors['diversity'] = diversity_score * 100
        
        # Academic performance
        academic_score = min(candidate.cgpa / 10.0, 1.0)
        factors['academic'] = academic_score * 100
        
        # Calculate weighted total
        total_score = (
            skills_score * self.weights['skills'] +
            location_score * self.weights['location'] +
            sector_score * self.weights['sector'] +
            qual_score * self.weights['qualification'] +
            diversity_score * self.weights['diversity'] +
            academic_score * self.weights['academic']
        ) * 100
        
        return min(total_score, 100.0), factors

class InternshipDatabase:
    """Database management for candidates and internships"""
    
    def __init__(self, db_path: str = "internship_system.db"):
        self.db_path = db_path
        self.init_database()
        self.load_sample_data()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Candidates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                skills TEXT NOT NULL,
                qualifications TEXT,
                location TEXT,
                sector TEXT,
                category TEXT,
                district TEXT,
                past_participation BOOLEAN,
                cgpa REAL,
                experience TEXT,
                email TEXT,
                phone TEXT
            )
        ''')
        
        # Internships table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS internships (
                id INTEGER PRIMARY KEY,
                company TEXT NOT NULL,
                title TEXT NOT NULL,
                required_skills TEXT NOT NULL,
                location TEXT,
                sector TEXT,
                capacity INTEGER,
                preferred_qualification TEXT,
                stipend INTEGER,
                duration TEXT,
                type TEXT,
                description TEXT
            )
        ''')
        
        # Matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id INTEGER,
                internship_id INTEGER,
                score REAL,
                factors TEXT,
                status TEXT,
                timestamp TEXT,
                FOREIGN KEY (candidate_id) REFERENCES candidates (id),
                FOREIGN KEY (internship_id) REFERENCES internships (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_sample_data(self):
        """Load sample data if database is empty"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if data already exists
        cursor.execute("SELECT COUNT(*) FROM candidates")
        if cursor.fetchone()[0] > 0:
            conn.close()
            return
        
        # Sample candidates
        sample_candidates = [
            Candidate(1, "Priya Sharma", ["Python", "Data Analysis", "SQL"], "B.Tech CSE", 
                     "Delhi", "Technology", "General", "New Delhi", False, 8.5, "Fresher", 
                     "priya@email.com", "+91-9999999999"),
            Candidate(2, "Rahul Kumar", ["Marketing", "Content Writing", "Social Media"], "MBA Marketing",
                     "Mumbai", "Marketing", "OBC", "Thane", False, 7.8, "1 year",
                     "rahul@email.com", "+91-8888888888"),
            Candidate(3, "Anjali Patel", ["Finance", "Excel", "Financial Modeling"], "B.Com",
                     "Ahmedabad", "Finance", "SC", "Sabarkantha", True, 8.2, "Fresher",
                     "anjali@email.com", "+91-7777777777"),
            Candidate(4, "Amit Singh", ["Java", "Spring Boot", "Microservices"], "MCA",
                     "Bangalore", "Technology", "General", "Rural Karnataka", False, 9.1, "2 years",
                     "amit@email.com", "+91-6666666666"),
            Candidate(5, "Sneha Reddy", ["UI/UX", "Figma", "User Research"], "B.Des",
                     "Hyderabad", "Design", "ST", "Adilabad", False, 8.7, "6 months",
                     "sneha@email.com", "+91-5555555555"),
        ]
        
        # Sample internships
        sample_internships = [
            Internship(1, "TechCorp India", "Data Science Intern", 
                      ["Python", "Data Analysis", "Machine Learning"], "Delhi", "Technology",
                      10, "B.Tech/MCA", 25000, "6 months", "Full-time",
                      "Work on ML projects and data analysis"),
            Internship(2, "MarketPro Solutions", "Digital Marketing Intern",
                      ["Marketing", "Content Writing", "Analytics"], "Mumbai", "Marketing",
                      5, "MBA/BBA", 20000, "3 months", "Part-time",
                      "Digital marketing campaigns and content creation"),
            Internship(3, "FinanceHub", "Financial Analyst Intern",
                      ["Finance", "Excel", "Financial Modeling"], "Ahmedabad", "Finance",
                      8, "B.Com/MBA Finance", 22000, "4 months", "Full-time",
                      "Financial analysis and modeling work"),
            Internship(4, "InnovateDesign", "UX Design Intern",
                      ["UI/UX", "Figma", "Prototyping"], "Hyderabad", "Design",
                      6, "B.Des/M.Des", 18000, "5 months", "Full-time",
                      "User experience design and prototyping"),
            Internship(5, "DevSolutions", "Backend Developer Intern",
                      ["Java", "Spring Boot", "Database"], "Bangalore", "Technology",
                      12, "B.Tech/MCA", 28000, "6 months", "Full-time",
                      "Backend development with Java and Spring Boot"),
        ]
        
        # Insert sample data
        for candidate in sample_candidates:
            self.add_candidate(candidate)
        
        for internship in sample_internships:
            self.add_internship(internship)
        
        conn.close()
    
    def add_candidate(self, candidate: Candidate):
        """Add a new candidate to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO candidates 
            (id, name, skills, qualifications, location, sector, category, district, 
             past_participation, cgpa, experience, email, phone)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (candidate.id, candidate.name, json.dumps(candidate.skills),
              candidate.qualifications, candidate.location, candidate.sector,
              candidate.category, candidate.district, candidate.past_participation,
              candidate.cgpa, candidate.experience, candidate.email, candidate.phone))
        
        conn.commit()
        conn.close()
    
    def add_internship(self, internship: Internship):
        """Add a new internship to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO internships 
            (id, company, title, required_skills, location, sector, capacity,
             preferred_qualification, stipend, duration, type, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (internship.id, internship.company, internship.title,
              json.dumps(internship.required_skills), internship.location,
              internship.sector, internship.capacity, internship.preferred_qualification,
              internship.stipend, internship.duration, internship.type,
              internship.description))
        
        conn.commit()
        conn.close()
    
    def get_all_candidates(self) -> List[Candidate]:
        """Retrieve all candidates from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM candidates")
        rows = cursor.fetchall()
        
        candidates = []
        for row in rows:
            candidate = Candidate(
                id=row[0], name=row[1], skills=json.loads(row[2]),
                qualifications=row[3], location=row[4], sector=row[5],
                category=row[6], district=row[7], past_participation=bool(row[8]),
                cgpa=row[9], experience=row[10], email=row[11], phone=row[12]
            )
            candidates.append(candidate)
        
        conn.close()
        return candidates
    
    def get_all_internships(self) -> List[Internship]:
        """Retrieve all internships from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM internships")
        rows = cursor.fetchall()
        
        internships = []
        for row in rows:
            internship = Internship(
                id=row[0], company=row[1], title=row[2],
                required_skills=json.loads(row[3]), location=row[4],
                sector=row[5], capacity=row[6], preferred_qualification=row[7],
                stipend=row[8], duration=row[9], type=row[10], description=row[11]
            )
            internships.append(internship)
        
        conn.close()
        return internships
    
    def save_matches(self, matches: List[Match]):
        """Save matching results to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clear previous matches
        cursor.execute("DELETE FROM matches")
        
        for match in matches:
            cursor.execute('''
                INSERT INTO matches 
                (candidate_id, internship_id, score, factors, status, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (match.candidate_id, match.internship_id, match.score,
                  json.dumps(match.factors), match.status, match.timestamp.isoformat()))
        
        conn.commit()
        conn.close()

class PMInternshipSystem:
    """Main system class coordinating all components"""
    
    def __init__(self):
        self.db = InternshipDatabase()
        self.matching_engine = AIMatchingEngine()
        self.matches = []
    
    def run_matching_algorithm(self) -> List[Match]:
        """Execute the AI matching algorithm"""
        print("🤖 Starting AI Matching Algorithm...")
        print("=" * 50)
        
        candidates = self.db.get_all_candidates()
        internships = self.db.get_all_internships()
        
        print(f"📊 Processing {len(candidates)} candidates and {len(internships)} internships")
        
        all_matches = []
        
        for candidate in candidates:
            candidate_matches = []
            
            for internship in internships:
                score, factors = self.matching_engine.calculate_match_score(candidate, internship)
                
                # Determine match status
                if score >= 70:
                    status = 'excellent'
                elif score >= 50:
                    status = 'good'
                else:
                    status = 'fair'
                
                match = Match(
                    candidate_id=candidate.id,
                    internship_id=internship.id,
                    score=round(score, 1),
                    factors=factors,
                    status=status,
                    timestamp=datetime.now()
                )
                
                candidate_matches.append(match)
            
            # Sort matches for this candidate by score and take top 3
            candidate_matches.sort(key=lambda x: x.score, reverse=True)
            all_matches.extend(candidate_matches[:3])
        
        # Sort all matches by score
        all_matches.sort(key=lambda x: x.score, reverse=True)
        
        # Save matches to database
        self.db.save_matches(all_matches)
        self.matches = all_matches
        
        print(f"✅ Generated {len(all_matches)} matches")
        print(f"🎯 Excellent matches: {len([m for m in all_matches if m.status == 'excellent'])}")
        print(f"👍 Good matches: {len([m for m in all_matches if m.status == 'good'])}")
        print(f"⚠️  Fair matches: {len([m for m in all_matches if m.status == 'fair'])}")
        
        return all_matches
    
    def display_top_matches(self, limit: int = 10):
        """Display top matching results"""
        if not self.matches:
            print("No matches available. Run the matching algorithm first.")
            return
        
        print("\n🏆 TOP MATCHES")
        print("=" * 80)
        
        candidates = {c.id: c for c in self.db.get_all_candidates()}
        internships = {i.id: i for i in self.db.get_all_internships()}
        
        for i, match in enumerate(self.matches[:limit], 1):
            candidate = candidates[match.candidate_id]
            internship = internships[match.internship_id]
            
            status_emoji = {"excellent": "🌟", "good": "👍", "fair": "⚠️"}
            
            print(f"\n{i}. {status_emoji[match.status]} MATCH SCORE: {match.score}%")
            print(f"   👤 {candidate.name} ({candidate.qualifications})")
            print(f"   🏢 {internship.title} at {internship.company}")
            print(f"   📍 {candidate.location} → {internship.location}")
            print(f"   💰 ₹{internship.stipend:,}/month • {internship.duration}")
            
            print(f"   📊 Score Breakdown:")
            for factor, score in match.factors.items():
                print(f"      • {factor.capitalize()}: {score:.1f}")
            print("-" * 60)
    
    def display_candidate_analytics(self):
        """Display candidate demographics and analytics"""
        candidates = self.db.get_all_candidates()
        
        print("\n📊 CANDIDATE ANALYTICS")
        print("=" * 40)
        
        # Category distribution
        category_dist = defaultdict(int)
        sector_dist = defaultdict(int)
        location_dist = defaultdict(int)
        
        for candidate in candidates:
            category_dist[candidate.category] += 1
            sector_dist[candidate.sector] += 1
            location_dist[candidate.location] += 1
        
        print("📋 Category Distribution:")
        for category, count in category_dist.items():
            print(f"   {category}: {count}")
        
        print("\n🏭 Sector Distribution:")
        for sector, count in sector_dist.items():
            print(f"   {sector}: {count}")
        
        print("\n📍 Location Distribution:")
        for location, count in location_dist.items():
            print(f"   {location}: {count}")
        
        # Average CGPA
        avg_cgpa = sum(c.cgpa for c in candidates) / len(candidates)
        print(f"\n📚 Average CGPA: {avg_cgpa:.2f}")
        
        # Past participation
        new_candidates = len([c for c in candidates if not c.past_participation])
        print(f"🆕 First-time candidates: {new_candidates}/{len(candidates)}")
    
    def display_internship_analytics(self):
        """Display internship opportunities analytics"""
        internships = self.db.get_all_internships()
        
        print("\n💼 INTERNSHIP ANALYTICS")
        print("=" * 40)
        
        sector_capacity = defaultdict(int)
        location_capacity = defaultdict(int)
        total_capacity = 0
        
        for internship in internships:
            sector_capacity[internship.sector] += internship.capacity
            location_capacity[internship.location] += internship.capacity
            total_capacity += internship.capacity
        
        print(f"📊 Total Capacity: {total_capacity} positions")
        
        print("\n🏭 Sector-wise Capacity:")
        for sector, capacity in sector_capacity.items():
            print(f"   {sector}: {capacity} positions")
        
        print("\n📍 Location-wise Capacity:")
        for location, capacity in location_capacity.items():
            print(f"   {location}: {capacity} positions")
        
        # Average stipend
        avg_stipend = sum(i.stipend for i in internships) / len(internships)
        print(f"\n💰 Average Stipend: ₹{avg_stipend:,.0f}/month")
    
    def export_results_to_csv(self, filename: str = "matching_results.csv"):
        """Export matching results to CSV"""
        if not self.matches:
            print("No matches to export. Run the matching algorithm first.")
            return
        
        candidates = {c.id: c for c in self.db.get_all_candidates()}
        internships = {i.id: i for i in self.db.get_all_internships()}
        
        export_data = []
        for match in self.matches:
            candidate = candidates[match.candidate_id]
            internship = internships[match.internship_id]
            
            row = {
                'Candidate_Name': candidate.name,
                'Candidate_Qualifications': candidate.qualifications,
                'Candidate_Location': candidate.location,
                'Candidate_Category': candidate.category,
                'Company': internship.company,
                'Internship_Title': internship.title,
                'Internship_Location': internship.location,
                'Stipend': internship.stipend,
                'Duration': internship.duration,
                'Match_Score': match.score,
                'Status': match.status,
                'Skills_Score': match.factors['skills'],
                'Location_Score': match.factors['location'],
                'Sector_Score': match.factors['sector'],
                'Diversity_Score': match.factors['diversity'],
                'Academic_Score': match.factors['academic']
            }
            export_data.append(row)
        
        df = pd.DataFrame(export_data)
        df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")

def main():
    """Main function with CLI interface"""
    system = PMInternshipSystem()
    
    print("🎯 PM INTERNSHIP AI MATCHING SYSTEM")
    print("=" * 50)
    print("Smart matching for better placements")
    print("=" * 50)
    
    while True:
        print("\n📋 MAIN MENU")
        print("1. Run AI Matching Algorithm")
        print("2. View Top Matches")
        print("3. Candidate Analytics")
        print("4. Internship Analytics")
        print("5. Export Results to CSV")
        print("6. Exit")
        
        choice = input("\n🔹 Select an option (1-6): ").strip()
        
        if choice == '1':
            system.run_matching_algorithm()
        
        elif choice == '2':
            limit = input("Number of top matches to display (default 10): ").strip()
            limit = int(limit) if limit.isdigit() else 10
            system.display_top_matches(limit)
        
        elif choice == '3':
            system.display_candidate_analytics()
        
        elif choice == '4':
            system.display_internship_analytics()
        
        elif choice == '5':
            filename = input("Enter filename (default: matching_results.csv): ").strip()
            filename = filename if filename else "matching_results.csv"
            system.export_results_to_csv(filename)
        
        elif choice == '6':
            print("👋 Thank you for using PM Internship AI Matching System!")
            break
        
        else:
            print("❌ Invalid option. Please select 1-6.")

if __name__ == "__main__":
    main()