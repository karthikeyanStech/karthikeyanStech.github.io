#!/usr/bin/env python3
"""
AI Construction Quantity Surveyor - Single File Implementation
==============================================================
Extracts BOQ and BBS from construction PDF drawings using Google Gemini AI

Author: Karthikeyan S (karthikeyanStech)
Date: 2025-01-04
Usage: python gemini_pdf_analyzer.py your_drawing.pdf
"""

import os
import sys
import json
import re
import time
from pathlib import Path

# Install required packages if not present
try:
    import google.generativeai as genai
    from tabulate import tabulate
except ImportError:
    print("📦 Installing required packages...")
    os.system("pip install google-generativeai tabulate -q")
    import google.generativeai as genai
    from tabulate import tabulate

# ============================================================================
# CONFIGURATION
# ============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-1.5-pro"

# ============================================================================
# MAIN ANALYZER CLASS
# ============================================================================

class QuantitySurveyorAI:
    """AI-powered construction quantity surveyor"""
    
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "❌ Gemini API key required!\n"
                "Get one from: https://makersuite.google.com/app/apikey\n"
                "Set it: export GEMINI_API_KEY='your-key-here'"
            )
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        print(f"✅ Gemini {GEMINI_MODEL} initialized!")
    
    def analyze_pdf(self, pdf_path: str) -> dict:
        """Analyze construction PDF and extract quantities"""
        
        print(f"\n{'='*80}")
        print(f"📄 ANALYZING: {pdf_path}")
        print(f"{'='*80}\n")
        
        # Upload PDF
        print("📤 Uploading PDF to Gemini...")
        uploaded_file = genai.upload_file(pdf_path)
        
        # Wait for processing
        while uploaded_file.state.name == "PROCESSING":
            print("⏳ Processing PDF...", end="\r")
            time.sleep(2)
            uploaded_file = genai.get_file(uploaded_file.name)
        
        if uploaded_file.state.name == "FAILED":
            raise ValueError("❌ PDF processing failed!")
        
        print("✅ PDF uploaded successfully!    \n")
        
        # Create comprehensive extraction prompt
        prompt = self._create_extraction_prompt()
        
        # Analyze with Gemini
        print("🤖 Analyzing with Gemini AI...\n")
        response = self.model.generate_content(
            [uploaded_file, prompt],
            request_options={"timeout": 600}
        )
        
        # Parse response
        result = self._parse_response(response.text)
        
        print("✅ Analysis complete!\n")
        return result
    
    def _create_extraction_prompt(self) -> str:
        """Create comprehensive extraction prompt"""
        return """You are an expert Construction Quantity Surveyor analyzing this PDF drawing.

EXTRACT ALL QUANTITIES FOR:

A. FOUNDATION
   - PCC bed: length × breadth × thickness → volume (m³)
   - RCC footing: length × breadth × depth → volume (m³)
   - Reinforcement bars: diameter (mm), spacing, count

B. SLAB
   - RCC slab: length × breadth × thickness → volume (m³)
   - Reinforcement: main & distribution bars

C. STAIRCASE
   - Dimensions: tread, riser, width
   - Number of steps
   - Concrete volume (m³)

D. ROOM AREAS
   - Each room: length × breadth → area (m²)

E. STEEL (BBS)
   For each member:
   - Bar diameter (mm)
   - Number of bars
   - Total length (m)
   - Weight (kg) = (diameter²/162) × total_length

RULES:
1. Detect scale (e.g., "1:100", "1\" = 8'0\"")
2. Convert ALL to METERS (1 ft = 0.3048 m, 1 in = 0.0254 m)
3. Extract from ALL views (plan, section, elevation, details)
4. State assumptions if dimensions missing

OUTPUT ONLY THIS JSON (no other text):

```json
{
  "scale": "1:100",
  "boq": [
    {
      "component": "Foundation F1",
      "type": "PCC Bed",
      "length": 2.0,
      "breadth": 2.0,
      "depth": 0.15,
      "quantity": 0.6,
      "unit": "m3",
      "source": "Foundation detail",
      "confidence": "High"
    },
    {
      "component": "Bedroom 1",
      "type": "Room Area",
      "length": 2.52,
      "breadth": 3.13,
      "depth": 0.0,
      "quantity": 7.89,
      "unit": "m2",
      "source": "Plan view",
      "confidence": "High"
    }
  ],
  "bbs": [
    {
      "mark": "F1-M1",
      "member": "Footing main",
      "dia_mm": 12,
      "count": 8,
      "length": 2.2,
      "total_length": 17.6,
      "weight_kg": 1.32,
      "notes": "12mm @ 200mm c/c"
    }
  ],
  "notes": [
    "Scale: 1:100",
    "All units in meters"
  ],
  "assumptions": [
    "Slab thickness 150mm assumed where not specified"
  ]
}
```

Extract NOW. Output ONLY valid JSON."""

    def _parse_response(self, response_text: str) -> dict:
        """Parse Gemini JSON response"""
        try:
            # Extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try direct JSON extraction
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON found in response")
            
            data = json.loads(json_str)
            
            # Recalculate steel weights for accuracy
            for bbs_item in data.get('bbs', []):
                dia = bbs_item['dia_mm']
                length = bbs_item['total_length']
                weight = (dia ** 2 / 162) * length * 1.02  # 2% wastage
                bbs_item['weight_kg'] = round(weight, 2)
            
            return data
            
        except Exception as e:
            print(f"⚠️  Parse error: {e}")
            print(f"\n{'='*80}")
            print("RAW RESPONSE:")
            print(f"{'='*80}")
            print(response_text)
            print(f"{'='*80}\n")
            return {
                "scale": "Unknown",
                "boq": [],
                "bbs": [],
                "notes": ["Parsing failed - see raw response above"],
                "assumptions": []
            }

# ============================================================================
# OUTPUT FORMATTING
# ============================================================================

def format_boq_table(boq_items: list) -> str:
    """Format BOQ as table"""
    if not boq_items:
        return "No BOQ items found."
    
    headers = ["Component", "Type", "L(m)", "B(m)", "D(m)", "Qty", "Unit", "Conf.", "Source"]
    data = [
        [
            item['component'],
            item['type'],
            f"{item['length']:.2f}",
            f"{item['breadth']:.2f}",
            f"{item['depth']:.2f}",
            f"{item['quantity']:.2f}",
            item['unit'],
            item['confidence'],
            item['source']
        ]
        for item in boq_items
    ]
    return tabulate(data, headers=headers, tablefmt="grid")

def format_bbs_table(bbs_items: list) -> str:
    """Format BBS as table"""
    if not bbs_items:
        return "No BBS items found."
    
    headers = ["Mark", "Member", "Dia(mm)", "Count", "Length(m)", "Total(m)", "Weight(kg)", "Notes"]
    data = []
    total_weight = 0
    
    for item in bbs_items:
        data.append([
            item['mark'],
            item['member'],
            item['dia_mm'],
            item['count'],
            f"{item['length']:.2f}",
            f"{item['total_length']:.2f}",
            f"{item['weight_kg']:.2f}",
            item['notes']
        ])
        total_weight += item['weight_kg']
    
    table = tabulate(data, headers=headers, tablefmt="grid")
    table += f"\n\n{'TOTAL STEEL:':<60} {total_weight:.2f} kg"
    return table

def save_results(result: dict, output_dir: str = "output"):
    """Save results to files"""
    Path(output_dir).mkdir(exist_ok=True)
    
    # JSON
    json_path = f"{output_dir}/result.json"
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ Saved: {json_path}")
    
    # Summary text
    summary_path = f"{output_dir}/summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CONSTRUCTION QUANTITY SURVEYOR - REPORT\n")
        f.write("Generated with Google Gemini AI\n")
        f.write(f"Author: Karthikeyan S (@karthikeyanStech)\n")
        f.write(f"Date: 2025-01-04\n")
        f.write("="*80 + "\n\n")
        f.write(f"Scale: {result.get('scale', 'Unknown')}\n\n")
        f.write("BILL OF QUANTITIES\n")
        f.write("-"*80 + "\n")
        f.write(format_boq_table(result.get('boq', [])))
        f.write("\n\n")
        f.write("BAR BENDING SCHEDULE\n")
        f.write("-"*80 + "\n")
        f.write(format_bbs_table(result.get('bbs', [])))
        f.write("\n\n")
        if result.get('notes'):
            f.write("NOTES:\n")
            for note in result.get('notes', []):
                f.write(f"• {note}\n")
        if result.get('assumptions'):
            f.write("\nASSUMPTIONS:\n")
            for assumption in result.get('assumptions', []):
                f.write(f"• {assumption}\n")
    print(f"✅ Saved: {summary_path}")
    
    # CSV for BOQ
    if result.get('boq'):
        csv_path = f"{output_dir}/boq.csv"
        with open(csv_path, 'w') as f:
            f.write("Component,Type,Length(m),Breadth(m),Depth(m),Quantity,Unit,Confidence,Source\n")
            for item in result['boq']:
                f.write(f"{item['component']},{item['type']},{item['length']},{item['breadth']},{item['depth']},{item['quantity']},{item['unit']},{item['confidence']},{item['source']}\n")
        print(f"✅ Saved: {csv_path}")
    
    # CSV for BBS
    if result.get('bbs'):
        csv_path = f"{output_dir}/bbs.csv"
        with open(csv_path, 'w') as f:
            f.write("Mark,Member,Diameter(mm),Count,Length(m),Total_Length(m),Weight(kg),Notes\n")
            for item in result['bbs']:
                f.write(f"{item['mark']},{item['member']},{item['dia_mm']},{item['count']},{item['length']},{item['total_length']},{item['weight_kg']},{item['notes']}\n")
        print(f"✅ Saved: {csv_path}")

def print_summary(result: dict):
    """Print summary statistics"""
    print("\n" + "="*80)
    print("📈 SUMMARY STATISTICS")
    print("="*80)
    
    # BOQ summary
    boq_items = result.get('boq', [])
    if boq_items:
        total_concrete = sum(item['quantity'] for item in boq_items if item['unit'] == 'm3')
        total_area = sum(item['quantity'] for item in boq_items if item['unit'] == 'm2')
        print(f"Total Concrete Volume: {total_concrete:.2f} m³")
        print(f"Total Floor Area: {total_area:.2f} m²")
        print(f"Total BOQ Items: {len(boq_items)}")
    
    # BBS summary
    bbs_items = result.get('bbs', [])
    if bbs_items:
        total_steel = sum(item['weight_kg'] for item in bbs_items)
        total_steel_length = sum(item['total_length'] for item in bbs_items)
        print(f"Total Steel Weight: {total_steel:.2f} kg")
        print(f"Total Steel Length: {total_steel_length:.2f} m")
        print(f"Total BBS Items: {len(bbs_items)}")
    
    print("="*80 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    
    print("\n" + "="*80)
    print("🏗️  AI CONSTRUCTION QUANTITY SURVEYOR")
    print("Powered by Google Gemini AI")
    print("Author: Karthikeyan S (@karthikeyanStech)")
    print("="*80 + "\n")
    
    # Get PDF path
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = input("📄 Enter PDF path: ").strip()
    
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return
    
    # Get API key
    api_key = GEMINI_API_KEY
    if not api_key:
        print("\n🔑 Gemini API Key Setup:")
        print("Get your key: https://makersuite.google.com/app/apikey")
        api_key = input("Enter API key: ").strip()
        if not api_key:
            print("❌ API key required!")
            return
    
    try:
        # Initialize analyzer
        analyzer = QuantitySurveyorAI(api_key)
        
        # Analyze PDF
        result = analyzer.analyze_pdf(pdf_path)
        
        # Display results
        print("="*80)
        print("📊 BILL OF QUANTITIES")
        print("="*80)
        print(format_boq_table(result.get('boq', [])))
        print()  
        
        print("="*80)
        print("🔩 BAR BENDING SCHEDULE")
        print("="*80)
        print(format_bbs_table(result.get('bbs', [])))
        print()  
        
        if result.get('notes'):
            print("="*80)
            print("📝 NOTES")
            print("="*80)
            for note in result['notes']:
                print(f"• {note}")
            print()  
        
        if result.get('assumptions'):
            print("="*80)
            print("⚠️  ASSUMPTIONS")
            print("="*80)
            for assumption in result['assumptions']:
                print(f"• {assumption}")
            print()  
        
        # Print summary
        print_summary(result)
        
        # Save option
        save = input("\n💾 Save results? (y/n): ").strip().lower()
        if save == 'y':
            output_dir = input("Output directory (default='output'): ").strip() or "output"
            save_results(result, output_dir)
        
        # JSON option
        show_json = input("\n📋 Show JSON? (y/n): ").strip().lower()
        if show_json == 'y':
            print("\n" + "="*80)
            print("JSON OUTPUT")
            print("="*80)
            print(json.dumps(result, indent=2))
        
        print("\n✨ Analysis complete!")
        print("Thank you for using AI Quantity Surveyor!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# DIRECT PYTHON IMPORT USAGE
# ============================================================================

def analyze_pdf_programmatically(pdf_path: str, api_key: str = None) -> dict:
    """
    Use this function when importing as a module
    
    Example:
        from gemini_pdf_analyzer import analyze_pdf_programmatically
        
        result = analyze_pdf_programmatically(
            pdf_path="my_drawing.pdf",
            api_key="your-api-key"
        )
        
        print(result['boq'])
        print(result['bbs'])
    """
    api_key = api_key or GEMINI_API_KEY
    analyzer = QuantitySurveyorAI(api_key)
    return analyzer.analyze_pdf(pdf_path)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()