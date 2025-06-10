"""Main runner for Task Management AI System"""

import os
import sys

# Import all weeks
import week1
import week2  
import week3
import week4

def main():
    print("="*60)
    print("TASK MANAGEMENT AI SYSTEM - COMPLETE PIPELINE")
    print("="*60)
    
    # Check if user wants to run specific week
    if len(sys.argv) > 1:
        week_num = sys.argv[1]
        if week_num == "1":
            print("\nRunning Week 1 only...")
            week1.main()  # Assuming each week file has a main() function
        elif week_num == "2":
            print("\nRunning Week 2 only...")
            week2.main()
        elif week_num == "3":
            print("\nRunning Week 3 only...")
            week3.main()
        elif week_num == "4":
            print("\nRunning Week 4 only...")
            week4.main()
        else:
            print("Invalid week number. Use 1, 2, 3, or 4")
    else:
        # Run all weeks in sequence
        print("\nRunning complete pipeline...\n")
        
        print("\n" + "-"*40)
        print("WEEK 1: Data Preprocessing")
        print("-"*40)
        week1.main()
        
        print("\n" + "-"*40)
        print("WEEK 2: Feature Extraction & Classification")
        print("-"*40)
        week2.main()
        
        print("\n" + "-"*40)
        print("WEEK 3: Advanced Models & Optimization")
        print("-"*40)
        week3.main()
        
        print("\n" + "-"*40)
        print("WEEK 4: Finalization & Results")
        print("-"*40)
        week4.main()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)

if __name__ == "__main__":
    main()
