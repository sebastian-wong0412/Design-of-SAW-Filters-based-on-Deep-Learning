import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx


# 1. 数据预处理
def preprocess_data(df, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(df.values)
    else:
        data_scaled = scaler.transform(df.values)
    return data_scaled, scaler


# 2. 创建序列数据
def create_sequences(data, n_steps_in, n_steps_out):
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:i + n_steps_in])
        y.append(data[i + n_steps_in:i + n_steps_in + n_steps_out])
    return np.array(X), np.array(y)


# 3. 构建LSTM模型 - 不考虑物理约束
def build_model(input_shape, output_shape, hyperparams):
    # 解析超参数
    lstm_units1 = int(hyperparams[0])
    lstm_units2 = int(hyperparams[1])
    learning_rate = hyperparams[2]

    model = Sequential()
    model.add(LSTM(lstm_units1, activation='relu', return_sequences=True, input_shape=input_shape))
    model.add(LSTM(lstm_units2, activation='relu'))
    model.add(Dense(output_shape))
    model.compile(optimizer=Adam(learning_rate), loss='mse')
    return model


# 4. PSO目标函数 - 评估模型性能
def pso_objective_function(X_train, y_train, X_val, y_val, input_shape, output_shape):
    def f(hyperparams):
        # 对每组粒子参数评估模型性能
        n_particles = hyperparams.shape[0]
        errors = []

        for i in range(n_particles):
            # 限制参数范围
            h = hyperparams[i].copy()
            h[0] = max(50, min(200, h[0]))  # LSTM1单元: 50-200
            h[1] = max(20, min(100, h[1]))  # LSTM2单元: 20-100
            h[2] = max(0.0001, min(0.01, h[2]))  # 学习率: 0.0001-0.01

            # 构建和训练模型
            model = build_model(input_shape, output_shape, h)
            model.fit(X_train, y_train.reshape(y_train.shape[0], -1),
                      epochs=20, verbose=0, batch_size=16)

            # 在验证集上评估
            y_pred = model.predict(X_val)
            mse = np.mean(np.square(y_pred - y_val.reshape(y_val.shape[0], -1)))
            errors.append(mse)

            # 清理，避免内存泄漏
            tf.keras.backend.clear_session()

        return np.array(errors)

    return f


# 5. 使用PSO优化模型超参数
def optimize_hyperparams_with_pso(X_train, y_train, X_val, y_val, input_shape, output_shape):
    # 定义搜索空间
    dimensions = 3  # LSTM1单元, LSTM2单元, 学习率
    bounds = ([50, 20, 0.0001], [200, 100, 0.01])  # 参数上下界
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

    # 创建优化器
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=dimensions, options=options, bounds=bounds)

    # 定义目标函数
    objective_func = pso_objective_function(X_train, y_train, X_val, y_val, input_shape, output_shape)

    # 运行优化
    best_cost, best_pos = optimizer.optimize(objective_func, iters=10, verbose=True)

    print(f"最佳超参数: LSTM1单元={int(best_pos[0])}, LSTM2单元={int(best_pos[1])}, 学习率={best_pos[2]}")
    print(f"PSO优化的最佳MSE: {best_cost}")
    return best_pos


# 计算性能评估指标
def calculate_metrics(y_true, y_pred):
    # 确保形状匹配
    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    # 计算MSE
    mse = mean_squared_error(y_true_flat, y_pred_flat)

    # 计算R²
    r2 = r2_score(y_true_flat, y_pred_flat)

    return mse, r2


# 6. 第一阶段：使用前4行预测5-8行
def first_stage_prediction(df, optimize=True):
    # 获取前4行数据和5-8行数据
    train_data = df.iloc[:4]
    val_data = df.iloc[4:8]

    # 预处理数据
    train_scaled, scaler = preprocess_data(train_data)
    val_scaled, _ = preprocess_data(val_data, scaler)

    # 创建序列
    n_steps_in, n_steps_out = 2, 2
    X_train, y_train = create_sequences(train_scaled, n_steps_in, n_steps_out)

    # 准备验证数据
    X_val = train_scaled[-n_steps_in:].reshape(1, n_steps_in, train_scaled.shape[1])
    y_val = val_scaled[:n_steps_out].reshape(1, n_steps_out, val_scaled.shape[1])

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = y_train.shape[1] * y_train.shape[2]

    if optimize:
        # 优化超参数
        best_hyperparams = optimize_hyperparams_with_pso(
            X_train, y_train, X_val, y_val, input_shape, output_shape)

        # 使用最优超参数构建模型
        model = build_model(input_shape, output_shape, best_hyperparams)
    else:
        # 使用默认超参数
        model = build_model(input_shape, output_shape, [100, 50, 0.001])

    # 训练最终模型
    model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=100, verbose=1)

    # 预测5-8行
    y_pred = model.predict(X_val)

    # 计算性能指标
    y_val_reshaped = y_val.reshape(y_val.shape[0], -1)
    mse, r2 = calculate_metrics(y_val_reshaped, y_pred)
    print(f"第一阶段预测性能 - MSE: {mse}, R²: {r2}")

    # 反转缩放
    y_pred = y_pred.reshape(1, n_steps_out, train_scaled.shape[1])
    pred_full = np.vstack([train_scaled, y_pred.reshape(n_steps_out, train_scaled.shape[1])])
    predictions = scaler.inverse_transform(pred_full)[4:8]

    # 计算原始尺度的性能指标
    original_y_val = val_data.values[:n_steps_out]
    original_y_pred = predictions[:n_steps_out]
    original_mse, original_r2 = calculate_metrics(original_y_val, original_y_pred)
    print(f"第一阶段原始尺度预测性能 - MSE: {original_mse}, R²: {original_r2}")

    return model, scaler, predictions


# 7. 第二阶段：使用1-8行预测第9行及以后
def second_stage_prediction(df, first_model, scaler, n_future=10, optimize=True):
    # 获取1-8行数据
    full_data = df.iloc[:8]

    # 如果有真实的未来数据可用于评估，则使用
    has_future_data = len(df) > 8
    if has_future_data:
        future_data = df.iloc[8:8 + n_future]

    # 预处理数据
    full_scaled, _ = preprocess_data(full_data, scaler)

    # 创建序列
    n_steps_in, n_steps_out = 4, 2
    X_full, y_full = create_sequences(full_scaled, n_steps_in, n_steps_out)

    input_shape = (X_full.shape[1], X_full.shape[2])
    output_shape = y_full.shape[1] * y_full.shape[2]

    if optimize:
        # 使用PSO优化迁移模型的超参数
        # 这里我们可以使用部分第一阶段的数据作为验证集
        X_val = X_full[-1:].copy()
        y_val = y_full[-1:].copy()

        best_hyperparams = optimize_hyperparams_with_pso(
            X_full[:-1], y_full[:-1], X_val, y_val, input_shape, output_shape)

        # 使用最优超参数构建模型
        transfer_model = build_model(input_shape, output_shape, best_hyperparams)
    else:
        # 使用与第一阶段模型相似的结构，但适应新的输入尺寸
        transfer_model = build_model(input_shape, output_shape, [100, 50, 0.001])

    # 微调模型
    transfer_model.fit(X_full, y_full.reshape(y_full.shape[0], -1), epochs=50, verbose=1)

    # 预测未来n_future个时间步
    future_predictions = []
    current_sequence = full_scaled[-n_steps_in:].reshape(1, n_steps_in, full_scaled.shape[1])

    for _ in range(n_future // n_steps_out):
        future = transfer_model.predict(current_sequence)
        future = future.reshape(1, n_steps_out, full_scaled.shape[1])
        future_predictions.append(future[0])
        # 更新序列
        current_sequence = np.vstack([current_sequence[0][n_steps_out:], future[0]])
        current_sequence = current_sequence.reshape(1, n_steps_in, full_scaled.shape[1])

    # 合并预测结果
    future_array = np.vstack(future_predictions)
    full_sequence = np.vstack([full_scaled, future_array])
    # 反转缩放
    predicted_values = scaler.inverse_transform(full_sequence)[8:]

    # 如果有真实的未来数据，计算性能指标
    if has_future_data and len(future_data) >= n_future:
        future_true = future_data.values[:n_future]
        future_pred = predicted_values[:n_future]
        future_mse, future_r2 = calculate_metrics(future_true, future_pred)
        print(f"第二阶段未来预测性能 - MSE: {future_mse}, R²: {future_r2}")

    return transfer_model, predicted_values


# 8. 用于新结构SAW谐振器的迁移学习
def transfer_to_new_structure(new_data, base_model, base_scaler, optimize=True):
    # 预处理新结构数据
    new_scaled, new_scaler = preprocess_data(new_data)

    # 创建序列
    n_steps_in, n_steps_out = 4, 2
    X_new, y_new = create_sequences(new_scaled, n_steps_in, n_steps_out)

    input_shape = (X_new.shape[1], X_new.shape[2])
    output_shape = y_new.shape[1] * y_new.shape[2]

    if optimize:
        # 为新结构优化超参数
        X_val = X_new[-1:].copy()
        y_val = y_new[-1:].copy()

        best_hyperparams = optimize_hyperparams_with_pso(
            X_new[:-1], y_new[:-1], X_val, y_val, input_shape, output_shape)

        # 使用最优超参数构建模型
        new_model = build_model(input_shape, output_shape, best_hyperparams)
    else:
        # 创建与基础模型结构相似的新模型
        new_model = build_model(input_shape, output_shape, [100, 50, 0.001])

    # 在新数据上微调
    new_model.fit(X_new, y_new.reshape(y_new.shape[0], -1), epochs=30, verbose=1)

    # 计算训练集上的性能
    y_pred_train = new_model.predict(X_new)
    train_mse, train_r2 = calculate_metrics(y_new.reshape(y_new.shape[0], -1), y_pred_train)
    print(f"新结构模型训练集性能 - MSE: {train_mse}, R²: {train_r2}")

    return new_model, new_scaler


# 使用示例
if __name__ == "__main__":
    # 使用pd.read_excel导入数据
    df = pd.read_csv('../data/data.csv')

    # 第一阶段：预测5-8行
    first_model, scaler, predictions_5_8 = first_stage_prediction(df, optimize=True)
    print("5-8行预测结果:", predictions_5_8)

    # 第二阶段：预测第9行及以后
    transfer_model, future_predictions = second_stage_prediction(
        df, first_model, scaler, n_future=10, optimize=True)
    print("未来预测结果:", future_predictions)

    # 泛化到新结构 - 如果有需要导入新结构数据，也使用Excel格式
    """
    new_df = pd.read_excel('XCUT.xlsx', engine='openpyxl')  # 可以根据需要修改文件名
    new_model, new_scaler = transfer_to_new_structure(
        new_df.iloc[:8], transfer_model, scaler, optimize=True)
    """
    print("使用PSO优化的非物理约束模型训练和预测流程完成")
