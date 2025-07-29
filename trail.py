from main import *
while True:
    x=input("Enter a question (or 'exit' to quit): ")
    if x.lower() == 'exit':
        break
    main(x)  # Call the main function to process the question
    print("\n" + "=" * 60 + "\n")  # Separator for clarity