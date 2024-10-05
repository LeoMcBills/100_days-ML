#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <cstring>

int main() {
    // Create the server socket
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "Failed to create server socket\n";
        return 1;
    }

    // Define the server address
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;  // Bind to all available interfaces
    address.sin_port = htons(8080);        // Use port 8080

    // Bind the socket to the specified address and port
    if (bind(server_fd, (struct sockaddr*)&address, addrlen) < 0) {
        std::cerr << "Bind failed\n";
        close(server_fd);
        return 1;
    }

    // Listen for incoming connections (max queue length = 3)
    if (listen(server_fd, 3) < 0) {
        std::cerr << "Listen failed\n";
        close(server_fd);
        return 1;
    }

    std::cout << "Server listening on port 8080...\n";

    // Accept a connection from a client
    int new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
    if (new_socket < 0) {
        std::cerr << "Accept failed\n";
        close(server_fd);
        return 1;
    }

    // Send a welcome message to the client
    const char* message = "Hello from server";
    send(new_socket, message, strlen(message), 0);
    std::cout << "Welcome message sent to client\n";

    return 0;
}