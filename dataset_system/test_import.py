from quick_flexible import SimpleFlexiblePredictor

predictor = SimpleFlexiblePredictor()
success, message = predictor.load_model()
print(f"Success: {success}")
print(f"Message: {message}")
