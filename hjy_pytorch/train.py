import torch
import time


# noinspection DuplicatedCode
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


def train_inception_net(net, loss_fun, optimizer, num_epochs,
                        train_iter, test_iter, device=torch.device('cpu')):
    """ 训练带辅助分类器的分类器 """
    net = net.to(device)
    print("\ntraining on ", device)
    print("start...")
    import time
    for epoch in range(num_epochs):
        net.training = True
        train_loss_sum, train_acc_num, num_train, start = 0., 0., 0, time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            outs = net(x)
            if net.aux_logits:
                loss = torch.tensor([0.]).to(device)
                for i in range(len(outs)):
                    if i == 0:
                        loss += loss_fun(outs[i], y)
                        train_loss_sum += loss.cpu().item()
                    else:
                        loss += loss_fun(outs[i], y) * 0.3
                train_acc_num += (outs[0].argmax(dim=1) == y).sum().cpu().item()
            else:
                loss = loss_fun(outs, y)
                train_loss_sum += loss.cpu().item()
                train_acc_num += (outs.argmax(dim=1) == y).sum().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            num_train += y.shape[0]
        train_loss_ave = train_loss_sum / num_train
        train_accuracy = train_acc_num / num_train

        # 测试样本集评估
        net.training = False
        test_acc_num, num_test = 0., 0
        for x_test, y_test in test_iter:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            test_acc_num += (net(x_test).argmax(dim=1) == y_test).sum().cpu().item()
            num_test += y_test.size()[0]
        test_accuracy = test_acc_num / num_test
        print("epoch:%d, train loss:%.4f, train accuracy:%.3f, test accuracy:%.3f, time:%.2fs"
              % (epoch + 1, train_loss_ave, train_accuracy, test_accuracy, time.time() - start))
    print("end...")
