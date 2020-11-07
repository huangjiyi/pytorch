import torch
import time


def train_classifier(net, loss_fun, optimizer, num_epochs,
                     train_iter, test_iter, device=torch.device('cpu')):
    """训练分类器"""
    net = net.to(device)  # 在GPU上训练网络
    print("\ntraining on ", device)
    print("start...\n")
    for epoch in range(num_epochs):
        train_loss_sum, train_acc_num, n, start = 0., 0., 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            loss = loss_fun(y_hat, y)
            optimizer.zero_grad()  # 梯度清零
            loss.backward()  # 梯度回传
            optimizer.step()  # 参数更新

            train_loss_sum += loss.cpu().item()
            train_acc_num += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.size()[0]
        train_loss_ave = train_loss_sum / n
        train_accuracy = train_acc_num / n

        # 测试样本集评估
        test_acc_num, n_test = 0., 0
        for X_test, y_test in test_iter:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_acc_num += (net(X_test).argmax(dim=1) == y_test).sum().cpu().item()
            n_test += y_test.size()[0]
        test_accuracy = test_acc_num / n_test
        print("epoch:%d, train loss:%.4f, train accuracy:%.3f, test accuracy:%.3f, time:%.2fs"
              % (epoch + 1, train_loss_ave, train_accuracy, test_accuracy, time.time() - start))
    print("\nend...")
