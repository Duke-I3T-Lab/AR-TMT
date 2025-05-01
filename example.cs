for (int i = 0; i < 10; i++)
{
    if (i == 5)
    {
        break; // Exit the loop when i equals 5
    }
    Debug.Log(i); // Prints 0, 1, 2, 3, 4
}
Debug.Log("Loop exited."); // This will execute after the loop

int number = 2;
switch (number)
{
    case 1:
        Debug.Log("One");
        break; // Exit the switch block
    case 2:
        Debug.Log("Two");
        break; // Exit the switch block
    default:
        Debug.Log("Default");
        break;
}
