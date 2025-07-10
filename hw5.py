import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Layer,Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Dropout ,LSTM, Input
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow.keras.backend as K
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error ,r2_score
import matplotlib.pyplot as plt
import time
import keras_tuner as kt
#a
# uploading data
cols = ['unit_number', 'time_in_cycles'] + \
       [f'op_setting_{i}' for i in range(1, 4)] + \
       [f'sensor_measurement_{i}' for i in range(1, 22)]
#train
train_FD1 = r'6. Turbofan Engine Degradation Simulation Data Set\train_FD001.txt'
train_df = pd.read_csv(train_FD1, sep=' ', header=None)
train_df.dropna(axis=1, inplace=True)
train_df.columns = cols
train_df.info()
#test
test_FD1 = r'6. Turbofan Engine Degradation Simulation Data Set\test_FD001.txt'
test_df = pd.read_csv(test_FD1, sep=' ', header=None)
test_df.dropna(axis=1, inplace=True)
test_df.columns = cols
#real_RUL
real_RUL = r'6. Turbofan Engine Degradation Simulation Data Set\RUL_FD001.txt'
rul_t = pd.read_csv(real_RUL, sep=' ', header=None)
rul_t = rul_t.dropna(axis=1)
rul_t.columns = ['RUL']

# calculate RUL
rul_df = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
rul_df.columns = ['unit_number', 'max_cycle']
train_df = train_df.merge(rul_df, on='unit_number')
train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
train_df.head()


max_cycles = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
max_cycles.columns = ['unit_number', 'max_cycle']
max_cycles['true_RUL'] = rul_t['RUL']
max_cycles['total_life'] = max_cycles['max_cycle'] + max_cycles['true_RUL']
test_df = test_df.merge(max_cycles[['unit_number', 'total_life']], on='unit_number')
test_df['RUL'] = test_df['total_life'] - test_df['time_in_cycles']

# ----------- 3. انتخاب ویژگی‌های مفید با حذف واریانس پایین -----------
sensor_cols = [col for col in train_df.columns if "sensor" in col]
selector = VarianceThreshold(threshold=0.01)
selector.fit(train_df[sensor_cols])
useful_sensor_cols = list(selector.get_feature_names_out(sensor_cols))

# حذف ویژگی‌های تکراری بر اساس همبستگی بالا (بیش از 0.98)
corr_matrix = train_df[useful_sensor_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.98)]
filtered_cols = [col for col in useful_sensor_cols if col not in to_drop]

cor_with_target = train_df[filtered_cols + ['RUL']].corr()['RUL'].abs().drop('RUL')
low_corr_features = cor_with_target[cor_with_target < 0.05].index.tolist()  
final_sensor_cols = [col for col in filtered_cols if col not in low_corr_features]

X_rf = train_df[final_sensor_cols]
y_rf = train_df['RUL']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)
importances = pd.Series(rf.feature_importances_, index=final_sensor_cols)
low_importance_features = importances[importances < 0.01].index.tolist()

selected_features = [col for col in final_sensor_cols if col not in low_importance_features]
train_df = train_df[selected_features + ['RUL', 'unit_number', 'time_in_cycles']]

# ----------- 4. نرمال‌سازی ویژگی‌ها -----------
scaler = MinMaxScaler()
train_df_scaled = train_df.copy()
train_df_scaled[selected_features] = scaler.fit_transform(train_df[selected_features])

test_df_scaled = test_df.copy()
test_df_scaled[selected_features] = scaler.transform(test_df[selected_features])

unique_units = train_df_scaled['unit_number'].unique()
train_units = unique_units[:int(0.7 * len(unique_units))]
val_units = unique_units[int(0.7 * len(unique_units)):]
# ----------- 6. آماده‌سازی داده‌ها با Sliding Window -----------
def prepare_windows(data, window_size=30):
    X, y = [], []
    for unit in data['unit_number'].unique():
        unit_data = data[data['unit_number'] == unit].sort_values('time_in_cycles')
        features = unit_data[selected_features].values
        rul = unit_data['RUL'].values
        for i in range(len(unit_data) - window_size + 1):
            X.append(features[i:i + window_size])
            y.append(rul[i + window_size - 1])
    return np.array(X), np.array(y)

X, y = prepare_windows(train_df_scaled, window_size=30)
X_test, y_test = prepare_windows(test_df_scaled)

# ----------- 7. تقسیم داده‌ها برای مدل LSTM -----------
X_train, y_train = prepare_windows(train_df_scaled[train_df_scaled['unit_number'].isin(train_units)])
X_val, y_val = prepare_windows(train_df_scaled[train_df_scaled['unit_number'].isin(val_units)])

# ----------- 5. محدود کردن RUL به 130 -----------
max_rul = 130  ###
train_df['RUL'] = train_df['RUL'].clip(upper=max_rul)
test_df['RUL'] = test_df['RUL'].clip(upper=max_rul)

# ----------- 8. آماده‌سازی داده‌ها برای مدل Dense -----------
model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  
])

model_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time_cnn = time.time()
history_cnn = model_cnn.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)
train_time_cnn = time.time() - start_time_cnn

y_pred_cnn = model_cnn.predict(X_test).flatten()

mae = mean_absolute_error(y_test, y_pred_cnn)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_cnn))
log_rmse = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred_cnn)))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"log-RMSE: {log_rmse:.2f}")

# ----------- 9. تعریف و آموزش مدل LSTM -----------
model_lstm = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1) 
])
model_lstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time_lstm = time.time()
history_lstm = model_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

train_time_lstm = time.time() - start_time_lstm

y_pred_lstm = model_lstm.predict(X_test).flatten()

mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, y_pred_lstm))
log_rmse_lstm = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred_lstm)))

print(f"LSTM MAE: {mae_lstm:.2f}")
print(f"LSTM RMSE: {rmse_lstm:.2f}")
print(f"LSTM log-RMSE: {log_rmse_lstm:.2f}")

#a-4
def build_cnn_model(hp):
    model = Sequential()
    
    model.add(Conv1D(
        filters=hp.Int('filters1', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size', [2, 3, 5]),
        activation='relu',
        padding='same',
        input_shape=(X.shape[1], X.shape[2])
    ))
    model.add(MaxPooling1D(pool_size=2))
    
    model.add(Conv1D(
        filters=hp.Int('filters2', min_value=32, max_value=128, step=32),
        kernel_size=3,
        activation='relu',
        padding='same'
    ))
    
    model.add(GlobalAveragePooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

    optimizer = Adam(learning_rate=lr) if optimizer_name == 'adam' else RMSprop(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model


def build_lstm_model(hp):
    model = Sequential()
    

    model.add(LSTM(
        units=hp.Int('lstm_units1', min_value=50, max_value=200, step=50),
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ))
    model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    
    model.add(LSTM(
        units=hp.Int('lstm_units2', min_value=50, max_value=150, step=50),
        return_sequences=False
    ))
    model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(64, activation='relu'))
    model.add(Dense(1)) 

    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop'])
    lr = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    optimizer = Adam(learning_rate=lr) if optimizer_choice == 'adam' else RMSprop(learning_rate=lr)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

tuner_cnn = kt.RandomSearch(
    build_cnn_model,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    directory='cnn_tuning',
    project_name='cnn_rul'
)

tuner_cnn.search_space_summary()

tuner_cnn.search(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    batch_size=64,  # مقدار ثابت برای batch size
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)],
    verbose=1
)

tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_loss',
    max_trials=10,  # تعداد تنظیمات مختلف
    executions_per_trial=1,
    directory='my_tuning',
    project_name='lstm_rul'
)

tuner.search_space_summary()

tuner.search(
    X_train, y_train,
    epochs=50,
    validation_data=(X_val, y_val),
    batch_size=64,
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
)


best_cnn_model = tuner_cnn.get_best_models(num_models=1)[0]
best_cnn_hps = tuner_cnn.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters for CNN:")
print(f"Filters1: {best_cnn_hps.get('filters1')}")
print(f"Filters2: {best_cnn_hps.get('filters2')}")
print(f"Kernel Size: {best_cnn_hps.get('kernel_size')}")
print(f"Dropout Rate: {best_cnn_hps.get('dropout_rate')}")
print(f"Optimizer: {best_cnn_hps.get('optimizer')}")
print(f"Learning Rate: {best_cnn_hps.get('learning_rate')}")


best_model = tuner.get_best_models(num_models=1)[0]
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best hyperparameters:")
print(f"LSTM Units 1: {best_hps.get('lstm_units1')}")
print(f"LSTM Units 2: {best_hps.get('lstm_units2')}")
print(f"Dropout1: {best_hps.get('dropout1')}")
print(f"Dropout2: {best_hps.get('dropout2')}")
print(f"Optimizer: {best_hps.get('optimizer')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")




# ----------- 10. تعریف و آموزش مدل cnn lstm -----------
model_cnn_lstm = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=(X.shape[1], X.shape[2])),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(1)  # خروجی RUL
])
model_cnn_lstm.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

start_time_cnn_lstm = time.time()
history_cnn_lstm = model_cnn_lstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)
train_time_cnn_lstm = time.time() - start_time_cnn_lstm

y_pred_cnn_lstm = model_cnn_lstm.predict(X_test).flatten()

mae_cnn_lstm = mean_absolute_error(y_test, y_pred_cnn_lstm)
rmse_cnn_lstm = np.sqrt(mean_squared_error(y_test, y_pred_cnn_lstm))
log_rmse_cnn_lstm = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred_cnn_lstm)))

print(f"CNN-LSTM MAE: {mae_cnn_lstm:.2f}")
print(f"CNN-LSTM RMSE: {rmse_cnn_lstm:.2f}")
print(f"CNN-LSTM log-RMSE: {log_rmse_cnn_lstm:.2f}")


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                 initializer="random_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                 initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

input_layer = Input(shape=(X.shape[1], X.shape[2]))
lstm_out = LSTM(100, return_sequences=True)(input_layer)
dropout = Dropout(0.2)(lstm_out)
attention_out = AttentionLayer()(dropout)
dense1 = Dense(64, activation='relu')(attention_out)
dropout2 = Dropout(0.2)(dense1)
output = Dense(1)(dropout2)

model_lstm_att = Model(inputs=input_layer, outputs=output)

model_lstm_att.compile(
    loss='mse',
    optimizer=Adam(learning_rate=0.001),
    metrics=['mae']
)

start_time_lstm_att = time.time()
history_lstm_att = model_lstm_att.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    verbose=1
)
train_time_lstm_att = time.time() - start_time_lstm_att

y_pred_att = model_lstm_att.predict(X_test).flatten()
mae_att = mean_absolute_error(y_test, y_pred_att)
rmse_att = np.sqrt(mean_squared_error(y_test, y_pred_att))
log_rmse_att = np.sqrt(mean_squared_error(np.log1p(y_test), np.log1p(y_pred_att)))

print(f"LSTM + Attention MAE: {mae_att:.2f}")
print(f"LSTM + Attention RMSE: {rmse_att:.2f}")
print(f"LSTM + Attention log-RMSE: {log_rmse_att:.2f}")

plt.plot(history_lstm_att.history['loss'], label='Train Loss')
plt.plot(history_lstm_att.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('LSTM + Attention Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

#result

def plot_loss(history, model_name):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} - Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history_cnn, "CNN")
plot_loss(history_lstm, "LSTM")
plot_loss(history_cnn_lstm, "CNN + LSTM")
plot_loss(history_lstm_att, "LSTM + Attention")

def plot_pred_vs_true(y_true, y_pred, model_name):
    plt.figure(figsize=(8,5))
    plt.scatter(range(len(y_true)), y_true, label='True RUL', alpha=0.6)
    plt.scatter(range(len(y_pred)), y_pred, label='Predicted RUL', alpha=0.6)
    plt.title(f'{model_name} - True vs Predicted RUL')
    plt.xlabel('Sample Index')
    plt.ylabel('RUL')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_pred_vs_true(y_test, y_pred_cnn, "CNN")
plot_pred_vs_true(y_test, y_pred_lstm, "LSTM")
plot_pred_vs_true(y_test, y_pred_cnn_lstm, "CNN + LSTM")
plot_pred_vs_true(y_test, y_pred_att, "LSTM + Attention")



def evaluate_model(y_true, y_pred, training_time):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    return {
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Training Time (s)': training_time,
        'MAPE (%)': mape
    }
results = []

results.append({'Model': 'CNN', **evaluate_model(y_test, y_pred_cnn, train_time_cnn)})
results.append({'Model': 'LSTM', **evaluate_model(y_test, y_pred_lstm, train_time_lstm)})
results.append({'Model': 'CNN + LSTM', **evaluate_model(y_test, y_pred_cnn_lstm, train_time_cnn_lstm)})
results.append({'Model': 'LSTM + Attention', **evaluate_model(y_test, y_pred_att, train_time_lstm_att)})

results_df = pd.DataFrame(results)
print(results_df)



def plot_cumulative_error(y_true, y_pred, model_name):
    errors = np.abs(y_true - y_pred)
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(errors)+1) / len(errors)
    plt.plot(sorted_errors, cumulative, label=model_name)

plt.figure(figsize=(8,5))
plot_cumulative_error(y_test, y_pred_cnn, "CNN")
plot_cumulative_error(y_test, y_pred_lstm, "LSTM")
plot_cumulative_error(y_test, y_pred_cnn_lstm, "CNN + LSTM")
plot_cumulative_error(y_test, y_pred_att, "LSTM + Attention")
plt.xlabel("Absolute Error")
plt.ylabel("Cumulative Proportion")
plt.title("Cumulative Error Curve")
plt.grid(True)
plt.legend()
plt.show()