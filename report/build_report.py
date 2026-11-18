# Script to build and compile the complete LaTeX report

import os
import subprocess
import sys

# Add builder directory to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'builder'))

from preamble import PREAMBLE
from front_matter import FRONT_MATTER
from ch1_introduction import CHAPTER_1
from ch2_literature_review import CHAPTER_2
from ch3_dataset import CHAPTER_3
from ch4_blackbox import CHAPTER_4
from ch5_methodology import CHAPTER_5
from ch6_software_architecture import CHAPTER_6
from ch7_experiments_results import CHAPTER_7
from ch8_case_studies import CHAPTER_8
from ch9_viva_defense import CHAPTER_9
from ch10_conclusion import CHAPTER_10
from references import REFERENCES

def main():
    tex_path = os.path.join(script_dir, "Glass_Box_Insight_Project_Report.tex")
    print(f"Assembling report to {tex_path}...")
    
    full_content = (
        PREAMBLE + "\n" +
        FRONT_MATTER + "\n" +
        CHAPTER_1 + "\n" +
        CHAPTER_2 + "\n" +
        CHAPTER_3 + "\n" +
        CHAPTER_4 + "\n" +
        CHAPTER_5 + "\n" +
        CHAPTER_6 + "\n" +
        CHAPTER_7 + "\n" +
        CHAPTER_8 + "\n" +
        CHAPTER_9 + "\n" +
        CHAPTER_10 + "\n" +
        REFERENCES
    )
    
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(full_content)
    print(f"Written {len(full_content)} characters to .tex file.")

if __name__ == "__main__":
    main()

