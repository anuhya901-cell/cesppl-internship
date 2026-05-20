import csv

class Worker:
    def __init__(self, name, employee_id, zone):
        self.name = name
        self.employee_id = employee_id
        self.zone = zone

    def display(self):
        print(f"{self.name} | {self.employee_id} | {self.zone}")


class Roster:
    def __init__(self):
        self.workers = []

    def add(self, worker):
        self.workers.append(worker)

    def remove(self, employee_id):
        self.workers = [
            worker for worker in self.workers
            if worker.employee_id != employee_id
        ]

    def find_by_name(self, name):
        for worker in self.workers:
            if worker.name.lower() == name.lower():
                return worker
        return None

    def save_to_csv(self, filename):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)

            writer.writerow(["Name", "Employee ID", "Zone"])

            for worker in self.workers:
                writer.writerow([
                    worker.name,
                    worker.employee_id,
                    worker.zone
                ])

        print(f"Data saved to {filename}")

worker1 = Worker("Anuhya", 101, "North")
worker2 = Worker("Rahul", 102, "South")
worker3 = Worker("Sneha", 103, "East")

roster = Roster()

roster.add(worker1)
roster.add(worker2)
roster.add(worker3)


found = roster.find_by_name("Rahul")

if found:
    print("Worker Found:")
    found.display()

roster.save_to_csv("workers.csv")