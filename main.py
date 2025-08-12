from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import requests
import json
import re

# Load environment variables
load_dotenv()
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
usda_api_key = os.getenv("USDA_API_KEY")

print("[DEBUG] DeepSeek Key:", deepseek_key[:8] + "..." if deepseek_key else "Not found")
print("[DEBUG] USDA Key:", usda_api_key[:8] + "..." if usda_api_key else "Not found")

# Initialize DeepSeek model
llm = ChatOpenAI(
    api_key=deepseek_key,
    base_url="https://api.deepseek.com/v1",
    model="deepseek-chat",
    temperature=0.2,
    max_tokens=1024
)

def parse_food_amount(user_input: str) -> list:
    """Parse food items and estimate portions from user input"""
    prompt = f"""
    You are a nutrition assistant that extracts food items and their quantities from text.
    Follow these rules:
    1. Identify all distinct food items in: "{user_input}"
    2. Standardize names to match USDA database (e.g., 'scrambled eggs' → 'egg, whole, raw')
    3. Estimate portions for vague descriptions:
       - 'a bowl' → '1 bowl (250 g)'
       - 'a few slices' → '3 slices (30 g)'
       - 'medium serving' → '1 medium serving (117 g)'
    4. Return ONLY a JSON array with format: [{{"food": "standardized name", "amount": "estimated quantity"}}]
    
    Example input: "I had 2 scrambled eggs and a cup of coffee"
    Example output: [{{"food": "egg, whole, raw", "amount": "2 large (100 g)"}}, {{"food": "coffee, brewed", "amount": "1 cup (240 ml)"}}]
    """
    
    try:
        response = llm.invoke(prompt)
        print("[DEBUG] Parsing response:", response.content)
        
        # Extract JSON from response if wrapped in markdown
        content = response.content
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()
        
        return json.loads(content)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"JSON parsing error: {str(e)}")
        return []
    except Exception as e:
        print(f"Parsing error: {str(e)}")
        return []

def query_usda(food_name: str) -> dict:
    """Query USDA FoodData Central API for nutrition information"""
    headers = {"Content-Type": "application/json"}
    params = {
        "api_key": usda_api_key,
        "query": food_name,
        "pageSize": 3,
        "dataType": ["Foundation", "SR Legacy"],
        "requireAllWords": "true"
    }
    
    try:
        resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            headers=headers,
            params=params,
            timeout=15
        )
        resp.raise_for_status()
        data = resp.json()
        
        if not data.get("foods"):
            print(f"No USDA data found for '{food_name}'")
            return {}
            
        # Select best match (highest score)
        best_match = max(data["foods"], key=lambda x: x.get("score", 0))
        return best_match
    except requests.exceptions.RequestException as e:
        print(f"USDA API error: {str(e)}")
        return {}
    except Exception as e:
        print(f"USDA processing error: {str(e)}")
        return {}

def extract_nutrients(usda_data: dict) -> dict:
    """Extract key nutrients from USDA API response"""
    nutrients = {}
    for nutrient in usda_data.get("foodNutrients", []):
        name = nutrient.get("nutrientName", "").lower()
        value = nutrient.get("value", 0)
        unit = nutrient.get("unitName", "").lower()
        
        # Match key nutrients with different naming variations
        if "energy" in name and "kcal" in unit:
            nutrients["calories"] = value
        elif "protein" in name:
            nutrients["protein"] = value
        elif "total lipid" in name or "total fat" in name:
            nutrients["fat"] = value
        elif "carbohydrate" in name and "by difference" in name:
            nutrients["carbohydrates"] = value
        elif "fiber" in name and "total dietary" in name:
            nutrients["fiber"] = value
        elif "sugars" in name and "total" in name:
            nutrients["sugars"] = value
        elif "fatty acids" in name and "saturated" in name:
            nutrients["saturated_fat"] = value
        elif "sodium" in name:
            nutrients["sodium"] = value
    
    return nutrients

def calculate_meal_totals(meal_info: list) -> dict:
    """Calculate total nutrition values for the meal"""
    totals = {
        "calories": 0,
        "protein": 0,
        "fat": 0,
        "carbohydrates": 0,
        "fiber": 0,
        "sugars": 0,
        "saturated_fat": 0,
        "sodium": 0
    }
    
    for item in meal_info:
        nutrients = item.get("nutrients", {})
        for key in totals:
            if key in nutrients:
                totals[key] += nutrients[key]
    
    return totals

def generate_analysis(meal_totals: dict) -> str:
    """Generate nutrition analysis and score using LLM"""
    nutrients = meal_totals
    prompt = f"""
    You are a professional nutritionist analyzing a meal. Use USDA Dietary Guidelines to evaluate:
    - Calories: {nutrients['calories']} kcal
    - Protein: {nutrients['protein']}g
    - Total Fat: {nutrients['fat']}g (Saturated: {nutrients.get('saturated_fat', 0)}g)
    - Carbohydrates: {nutrients['carbohydrates']}g
    - Fiber: {nutrients['fiber']}g
    - Sugars: {nutrients.get('sugars', 0)}g
    - Sodium: {nutrients.get('sodium', 0)}mg
    
    Provide a comprehensive analysis with:
    1. Nutrition Score: X/10 (0-10 scale based on USDA guidelines)
    2. Key Observations: Highlight strengths and weaknesses
    3. Macronutrient Balance: Evaluate protein/fat/carb ratios
    4. Micronutrient Note: Comment on vitamin/mineral content
    5. Improvement Suggestions: Specific, actionable advice
    6. Final Verdict: Overall assessment
    
    Structure your response with clear sections:
    **Score: X/10**
    **Breakdown:**
    - [Key observations]
    **Pros:**
    - [Strengths]
    **Cons:**
    - [Weaknesses]
    **Suggestions:**
    - [Actionable advice]
    **Verdict:**
    - [Overall assessment]
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"Analysis generation error: {str(e)}")
        return "Unable to generate nutrition analysis at this time."

def nutrition_agent(user_input: str) -> str:
    """Main agent function to process meal input"""
    print(f"[DEBUG] User input: {user_input}")
    
    # Parse food items from input
    parsed_foods = parse_food_amount(user_input)
    if not parsed_foods:
        return "❌ Could not identify any foods in your input. Please try again with more specific descriptions."
    
    print(f"[DEBUG] Parsed foods: {json.dumps(parsed_foods, indent=2)}")
    
    # Retrieve nutrition data for each food
    meal_info = []
    for item in parsed_foods:
        food_name = item.get("food", "")
        amount = item.get("amount", "")
        
        if not food_name:
            continue
            
        usda_data = query_usda(food_name)
        if not usda_data:
            print(f"Skipping {food_name} - no USDA data")
            continue
            
        nutrients = extract_nutrients(usda_data)
        meal_info.append({
            "food": food_name,
            "amount": amount,
            "nutrients": nutrients
        })
        
        print(f"[DEBUG] {food_name} nutrients: {json.dumps(nutrients, indent=2)}")
    
    if not meal_info:
        return "⚠️ No nutritional data found for the foods mentioned. Try different food names."
    
    # Calculate meal totals
    meal_totals = calculate_meal_totals(meal_info)
    print(f"[DEBUG] Meal totals: {json.dumps(meal_totals, indent=2)}")
    
    # Generate analysis
    return generate_analysis(meal_totals)

if __name__ == "__main__":
    print("[AGENT] Please enter what you ate in this meal (e.g., 'I ate 2 eggs and a bowl of rice'):")
    user_input = input().strip()
    
    if not user_input:
        print("No input received. Exiting.")
        exit()
    
    result = nutrition_agent(user_input)
    print("\n[AGENT] Nutrition analysis and score:")
    print(result)