#!/usr/bin/env python3
"""
Setup script to initialize Git repository and prepare for GitHub
"""

import os
import subprocess
import sys

def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_git_installed():
    """Check if Git is installed."""
    try:
        subprocess.run(["git", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def setup_git_repository():
    """Initialize Git repository and make initial commit."""
    if not check_git_installed():
        print("❌ Git is not installed. Please install Git first.")
        return False
    
    # Check if already a Git repository
    if os.path.exists(".git"):
        print("ℹ️  Git repository already exists")
        return True
    
    # Initialize Git repository
    if not run_command("git init", "Initializing Git repository"):
        return False
    
    # Add all files
    if not run_command("git add .", "Adding files to Git"):
        return False
    
    # Make initial commit
    if not run_command('git commit -m "Initial commit: Breast Cancer Detection project"', "Making initial commit"):
        return False
    
    return True

def print_github_instructions():
    """Print instructions for setting up GitHub repository."""
    print("\n" + "="*60)
    print("🚀 GITHUB SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1. Create a new repository on GitHub:")
    print("   - Go to https://github.com/new")
    print("   - Repository name: breast-cancer-detection")
    print("   - Description: Machine learning project for breast cancer detection")
    print("   - Make it Public or Private (your choice)")
    print("   - Don't initialize with README (we already have one)")
    print("   - Click 'Create repository'")
    
    print("\n2. Connect your local repository to GitHub:")
    print("   Replace 'yourusername' with your actual GitHub username:")
    print("   git remote add origin https://github.com/yourusername/breast-cancer-detection.git")
    
    print("\n3. Push your code to GitHub:")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    print("\n4. Verify your repository:")
    print("   - Go to your GitHub repository URL")
    print("   - Check that all files are uploaded")
    print("   - Verify the README is displayed correctly")
    
    print("\n5. Optional: Set up GitHub Pages (for project website):")
    print("   - Go to Settings > Pages")
    print("   - Source: Deploy from a branch")
    print("   - Branch: main, folder: / (root)")
    print("   - Save")

def print_project_summary():
    """Print a summary of the project files."""
    print("\n" + "="*60)
    print("📁 PROJECT FILES SUMMARY")
    print("="*60)
    
    files = [
        ("README.md", "Comprehensive project documentation"),
        ("requirements.txt", "Python dependencies"),
        ("LICENSE", "MIT license file"),
        (".gitignore", "Git ignore patterns"),
        ("breast_cancer_analysis.py", "Main analysis script"),
        ("test_project.py", "Test suite"),
        ("setup.py", "Package setup configuration"),
        ("CONTRIBUTING.md", "Contribution guidelines"),
        ("BreastCancerDetection.ipynb", "Original Jupyter notebook"),
        ("dataset.csv", "Breast cancer dataset"),
        (".github/workflows/test.yml", "GitHub Actions CI/CD")
    ]
    
    for filename, description in files:
        status = "✅" if os.path.exists(filename) else "❌"
        print(f"{status} {filename:<30} - {description}")

def main():
    """Main function to set up the project."""
    print("Breast Cancer Detection - GitHub Setup")
    print("="*50)
    
    # Print project summary
    print_project_summary()
    
    # Setup Git repository
    if setup_git_repository():
        print("\n✅ Git repository setup completed!")
    else:
        print("\n❌ Git repository setup failed!")
        return
    
    # Print GitHub instructions
    print_github_instructions()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED!")
    print("="*60)
    print("\nYour project is now ready to be pushed to GitHub!")
    print("Follow the instructions above to complete the setup.")

if __name__ == "__main__":
    main()
