import argparse
from FederatedPreparer import FederatedPreparer
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import traceback

class Federated:
    """
    Prepare the Federated Learning and launch it in headless mode on Firefox
    Server has to be launched previously
    
    Attributs:
    nb_users -- Number of users for the Federated Learning
    batch_size -- Size of the subset to attribute to one client 
    communication_round -- Number of model updates
    federated_preparer -- Instance of the FederatedPreparer class 
    url -- URL of the web application
    """
    
    def __init__(self,args):
        """
        Initializes a new instance of the Federated class
        
        Arguments:
        args -- arguments passed at the execution of the script
        """
        
        self.nb_users = args.nb_users
        self.batch_size = args.batch_size
        self.communication_round = args.nb_roc
        self.federated_preparer = FederatedPreparer(self.nb_users,self.batch_size)
        self.url = 'http://localhost:8080'
        
        
    def launch_federated_headless(self, url):
        """
        Connects to the web web page and launches the simulation by clicking 
        on the launch federated button
        
        Arguments:
        url -- URL of the web application
        """

        options = Options()
        options.add_argument('--headless')  # Run in headless mode for no UI
        service = Service('/usr/local/bin/geckodriver')  # Update this path to your WebDriver
        driver = webdriver.Firefox(service=service, options=options)


        try:
            # Open the web application
            driver.get(url)
            print("Page loaded successfully")

            # Wait until the page is loaded and a specific element is present
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            print("Body tag found")

            # Log current page source (for debugging purposes)
            with open("page_source.html", "w") as f:
                f.write(driver.page_source)

            # Fill out input fields
            print("Attempting to find the 'nb_users' field")
            user_field = driver.find_element(By.ID, 'nb_users')
            user_field.clear()
            user_field.send_keys(str(self.nb_users))
            print(f"'nb_users' field found and set to {self.nb_users}")

            print("Attempting to find the 'nb_roc' field")
            communication_round_field = driver.find_element(By.ID, 'nb_roc')
            communication_round_field.clear()
            communication_round_field.send_keys(str(self.communication_round))
            print(f"'nb_roc' field found and set to {self.communication_round}")

            # Click the launch button
            print("Attempting to find the 'launch_federated' button")
            button = driver.find_element(By.ID, 'launch_federated')
            print("Button found")
            button.click()
            print("Launch button clicked")

            # Polling to check for the completion element
            max_wait_time = 8 * 60 * 60  # 8 hours in seconds
            polling_interval = 5 * 60  # Check every 5 minutes
            elapsed_time = 0

            while elapsed_time < max_wait_time:
                try:
                    print("Checking for 'completion_element_id'")
                    if driver.find_element(By.ID, 'completion_element_id'):
                        print("Training Completed")
                        break
                except Exception as e:
                    print(f"Element not found, retrying... ({elapsed_time}s elapsed)")

                # Sleep for the polling interval and update elapsed time
                time.sleep(polling_interval)
                elapsed_time += polling_interval

            if elapsed_time >= max_wait_time:
                print("Timeout: Training did not complete within the expected time")

        except Exception as e:
            print(f"Error: {e}")
            print(traceback.format_exc())  # Print the full stack trace for debugging
            # Optional: Save the current page state to a file for analysis
            with open("error_page_source.html", "w") as f:
                f.write(driver.page_source)
            driver.save_screenshot("error_screenshot.png")  # Save screenshot for analysis

        finally:
            # Clean up and close the browser
            driver.quit()
            
        
    def __call__(self):
        """Prepare and launch the Federated Learning"""
        self.federated_preparer()
        self.launch_federated_headless(self.url)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="Launch Federated"
    )
    
    parser.add_argument("--nb_users", type=int, help="Number of users")
    parser.add_argument("--batch_size", type=int, help="Size of user batch of images")
    parser.add_argument("--nb_roc", type=str, help="Number of round of communication")
    
    args = parser.parse_args()
    
    obj = Federated(args)
    obj()