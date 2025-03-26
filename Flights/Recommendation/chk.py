def recommend_best_flights(departure, arrival, date, df, model, scaler):
    # Convert user input to encoded format
    if departure in label_encoders['Departure'].classes_ and arrival in label_encoders['Arrival'].classes_:
        dep_enc = label_encoders['Departure'].transform([departure])[0]
        arr_enc = label_encoders['Arrival'].transform([arrival])[0]
    else:
        print("Invalid Departure or Arrival location")
        return None
    
    # Filter available flights for the given date
    date = pd.to_datetime(date)
    available_flights = df[(df['Departure'] == dep_enc) & (df['Arrival'] == arr_enc) &
                           (df['Departure_date'] >= date - pd.Timedelta(days=1)) &
                           (df['Departure_date'] <= date + pd.Timedelta(days=1))]
    
    if available_flights.empty:
        print("No flights found for the given criteria.")
        return None
    
    # Prepare real-time flights for prediction
    available_features = available_flights[features]
    available_features[['Calculated_Flight_Price', 'Duration_minutes', 'Distance_km']] = scaler.transform(available_features[['Calculated_Flight_Price', 'Duration_minutes', 'Distance_km']])
    available_flights['Predicted_Score'] = model.predict(available_features)
    
    # Return top 5 recommended flights
    return available_flights.sort_values(by='Predicted_Score', ascending=True).head(5)

# Example usage
departure_input = "Coimbatore International Airport (CJB)"
arrival_input = "Visakhapatnam Airport (VTZ)"
date_input = "2025-04-10"
recommendations = recommend_best_flights(departure_input, arrival_input, date_input, df, model, scaler)
print(recommendations)
# print(df)