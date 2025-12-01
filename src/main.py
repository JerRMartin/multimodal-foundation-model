"""Main module for the src package."""
from src.head_gesture_detector import HeadGestureDetector
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Head gesture detection system")
    parser.add_argument('--control', action='store_true', 
                       help="Enable Xbox controller and haptic feedback")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create detector with control setting from command line
    detector = HeadGestureDetector(control=args.control)
    detector.run()