import keras
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D

# Define your model architecture
def create_model(dropoutRate=0.5, kernels=1, kernLength=32, F1=8, D=2, F2=16, dropoutType='Dropout', loss='categorical_crossentropy', optimizer='adam'):
    model = Sequential()
    model.add(Conv1D(filters=F1, kernel_size=kernLength, activation='relu', input_shape=(input_shape)))
    model.add(Dropout(dropoutRate, noise_shape=None, seed=None))
    model.add(Conv1D(filters=F2, kernel_size=kernLength, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# Define hyperparameters grid
param_grid = {
    'dropoutRate': [0.5, 0.6],  # Example values, you can expand this list
    'kernels': [1, 2],
    'kernLength': [32, 64],
    'F1': [8, 16],
    'D': [2, 4],
    'F2': [16, 32],
    'dropoutType': ['Dropout', 'SpatialDropout1D'],
    'loss': ['categorical_crossentropy'],
    'optimizer': ['adam'],
    'batch_size': [16],
    'epochs': [30]
}

# Create a model
model = KerasClassifier(build_fn=create_model, input_shape=(your_input_shape), output_shape=(your_output_shape), verbose=0)
model = keras.models.Model.
# Perform grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)  # You can adjust the number of cross-validation folds
grid_result = grid.fit(X_train, y_train)  # X_train and y_train are your training data and labels

# Print the best parameters and corresponding accuracy
print("Best parameters found: ", grid_result.best_params_)
print("Best accuracy found: ", grid_result.best_score_)
