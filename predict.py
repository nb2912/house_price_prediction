from model import HousePricePredictor
import pandas as pd

def get_numeric_input(prompt, min_value=0):
    while True:
        try:
            value = float(input(prompt))
            if value >= min_value:
                return value
            print(f"Please enter a value greater than or equal to {min_value}")
        except ValueError:
            print("Please enter a valid number")

def get_yes_no_input(prompt):
    while True:
        response = input(prompt + " (yes/no): ").lower()
        if response in ['yes', 'y']:
            return 1
        elif response in ['no', 'n']:
            return 0
        print("Please enter 'yes' or 'no'")

def get_choice_input(prompt, choices):
    print(prompt)
    for i, choice in enumerate(choices, 1):
        print(f"{i}: {choice}")
    
    while True:
        try:
            choice = int(input("Enter your choice (number): "))
            if 1 <= choice <= len(choices):
                return choice - 1
            print(f"Please enter a number between 1 and {len(choices)}")
        except ValueError:
            print("Please enter a valid number")

def main():
    print("Welcome to House Price Predictor!")
    print("=================================")
    
    # Initialize and train the model
    predictor = HousePricePredictor()
    print("\nLoading and training the model...")
    predictor.load_data('data/house_cleaned.csv')
    predictor.preprocess_data()
    predictor.prepare_features()
    predictor.train_model()
    
    # Get location data
    locations = predictor.get_unique_locations()
    
    while True:
        print("\nPlease enter the details of the house:")
        print("--------------------------------------")
        
        # Get location details
        print("\nLocation Details:")
        state_idx = get_choice_input("\nSelect State:", locations['states'])
        state = locations['states'][state_idx]
        
        # Filter cities based on selected state
        state_cities = sorted(set(predictor.data[predictor.data['state'] == state]['city'].unique()))
        city_idx = get_choice_input("\nSelect City:", state_cities)
        city = state_cities[city_idx]
        
        # Filter localities based on selected city
        city_localities = sorted(set(predictor.data[predictor.data['city'] == city]['locality'].unique()))
        locality_idx = get_choice_input("\nSelect Locality:", city_localities)
        locality = city_localities[locality_idx]
        
        # Get pincode
        pincode = input("\nEnter Pincode (6 digits, press Enter to skip): ").strip()
        if pincode and (not pincode.isdigit() or len(pincode) != 6):
            print("Invalid pincode format. Skipping pincode.")
            pincode = ""
        
        # Get basic property details
        print("\nBasic Property Details:")
        area = get_numeric_input("Enter area in square feet: ")
        bedrooms = get_numeric_input("Enter number of bedrooms: ")
        bathrooms = get_numeric_input("Enter number of bathrooms: ")
        price_per_sqft = get_numeric_input("Enter expected price per square feet: ")
        floor_num = get_numeric_input("Enter floor number (0 for ground floor): ")
        
        # Get property age
        print("\nProperty Age Categories:")
        print("1: 0 to 6 months old")
        print("2: 6 months to 1 year old")
        print("3: 1 to 5 years old")
        print("4: 5 to 10 years old")
        print("5: 10+ years old")
        
        age_choice = get_choice_input("Select property age:", [
            "0 to 6 months old",
            "6 months to 1 year old",
            "1 to 5 years old",
            "5 to 10 years old",
            "10+ years old"
        ])
        
        age_mapping = {
            0: 0.25,  # 0-6 months
            1: 0.5,   # 6-12 months
            2: 3,     # 1-5 years
            3: 7.5,   # 5-10 years
            4: 12     # 10+ years
        }
        property_age = age_mapping[age_choice]
        
        # Get amenities
        print("\nAmenities:")
        has_gym = get_yes_no_input("Does it have a gym/fitness centre?")
        has_pool = get_yes_no_input("Does it have a swimming pool?")
        has_security = get_yes_no_input("Does it have security personnel?")
        has_park = get_yes_no_input("Does it have a park?")
        
        # Property type
        property_type_idx = get_choice_input("\nSelect Property Type:", [
            "Apartment",
            "Independent House",
            "Builder Floor"
        ])
        
        # Facing direction
        facing_idx = get_choice_input("\nSelect Facing Direction:", [
            "North",
            "South",
            "East",
            "West",
            "North-East",
            "North-West",
            "South-East",
            "South-West"
        ])
        
        # Create features dictionary
        features = {
            'area': area,
            'bedRoom': bedrooms,
            'bathroom': bathrooms,
            'price_per_sqft': price_per_sqft,
            'has_gym': has_gym,
            'has_pool': has_pool,
            'has_security': has_security,
            'has_park': has_park,
            'floor_num': floor_num,
            'property_age': property_age,
            'property_type_encoded': property_type_idx,
            'facing_encoded': facing_idx,
            'state_encoded': state_idx,
            'city_encoded': city_idx,
            'locality_encoded': locality_idx
        }
        
        # Get prediction
        predicted_price = predictor.predict_price(features)
        
        print("\nProperty Details Summary:")
        print("========================")
        print(f"Location: {locality}, {city}, {state}")
        if pincode:
            print(f"Pincode: {pincode}")
        print(f"Area: {area} sq ft")
        print(f"Bedrooms: {bedrooms}")
        print(f"Bathrooms: {bathrooms}")
        print(f"Floor: {floor_num}")
        print(f"Property Type: {['Apartment', 'Independent House', 'Builder Floor'][property_type_idx]}")
        print(f"Facing: {['North', 'South', 'East', 'West', 'North-East', 'North-West', 'South-East', 'South-West'][facing_idx]}")
        
        print("\nPrediction Results:")
        print("==================")
        print(f"Predicted Price: ₹{predicted_price:.2f} Crores")
        print(f"Predicted Price in Lakhs: ₹{predicted_price*100:.2f} Lakhs")
        
        # Ask if user wants to make another prediction
        if input("\nWould you like to make another prediction? (yes/no): ").lower() not in ['yes', 'y']:
            break
    
    print("\nThank you for using House Price Predictor!")

if __name__ == "__main__":
    main() 