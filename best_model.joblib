import joblib

best_model_name = None
best_mse = float('inf')

for model_name, performance in model_performance.items():
    # Access the best_score from the GridSearchCV object
    current_mse = performance['Best MSE'].best_score_
    if current_mse < best_mse:
        best_mse = current_mse
        best_model_name = model_name

# Get the best model object from the GridSearchCV result
best_grid_search = model_performance[best_model_name]['Best Model']
best_model = best_grid_search.best_estimator_

# Save the best performing model
filename = 'best_model.joblib'
joblib.dump(best_model, filename)

print(f"Best performing model is {best_model_name} with MSE: {best_mse}")
print(f"Saved best model to {filename}")
