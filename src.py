def model_results(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print('Accuracy: '+ str(model.score(X_test,y_test)))
    print(plot_confusion_matrix(model,X_test,y_test,display_labels=['Non-Fraudulent','Fraudulent'],values_format='.5g'))
    print(plot_precision_recall_curve(model, X_test, y_test))